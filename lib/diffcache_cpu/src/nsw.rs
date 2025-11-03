use crate::{Distance, HugeArray, L2Square, Neighbour, NodeID};
use std::{array, cmp::{min, Reverse}, collections::BinaryHeap, intrinsics::{prefetch_read_data, sqrtf32}, io::{BufReader, BufWriter}, iter::repeat_with, marker::PhantomData, path::Path, sync::atomic::Ordering, time::Instant};
use crate::{Value};
use std::cell::RefCell;
use super::tlset::TLSet;
use ndarray::{s, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use std::sync::atomic::AtomicUsize;
use crate::InnerProduct;

const MAX_NUM_NEIGHBOURS: usize = 32;

pub const BATCH_SIZE: usize = 1;

#[derive(Serialize, Deserialize, Clone)]
pub struct Options {
    pub m: usize,          // number of neighbors to select
    pub ef_cons: usize,
    pub n_max: usize,      // maximum number of nodes
    pub r_sq: f32,
}

#[derive(Clone, Debug)]
pub struct NodeInfo<T: Clone> {
    pub node_id: NodeID,
    pub vec: Vec<T>
}

#[repr(C)]
struct Node<T: Clone> {
    valid: bool,                              // 1B
    edges_len: u8,                            // 1B
    edges: [NodeID; MAX_NUM_NEIGHBOURS],      // 128B (4B * 32)
    phantom: PhantomData<T>
}
pub struct Statistic {
    pub num_searches: AtomicUsize,
    pub search_latency: AtomicUsize,
    pub num_expanded_nodes: AtomicUsize,
    pub num_candidates_raw: AtomicUsize,
    pub num_candidates_unvisited: AtomicUsize,
    pub num_candidates_pruned: AtomicUsize,
}

impl Statistic {
    fn new() -> Self {
        Statistic {
            num_searches: AtomicUsize::new(0),
            search_latency: AtomicUsize::new(0),
            num_expanded_nodes: AtomicUsize::new(0),
            num_candidates_raw: AtomicUsize::new(0),
            num_candidates_unvisited: AtomicUsize::new(0),
            num_candidates_pruned: AtomicUsize::new(0),
        }
    }
}

pub struct Index<T: Clone> {
    nodes: HugeArray<Node<T>>,
    keys: HugeArray<T>,
    vals: HugeArray<T>,
    options: Options,
    m_max: usize,
    dim: usize,
    visited_set: TLSet,
    pub stats: Statistic
}

#[derive(Serialize, Deserialize, Clone)]
struct NodeSnapshot {
    id: NodeID,
    edges: Vec<NodeID>
}

#[derive(Serialize, Deserialize, Clone)]
struct IndexSnapshot {
    nodes: Vec<NodeSnapshot>,
    keys: Vec<f32>,
    vals: Vec<f32>,
    options: Options,
    dim: usize,
}

impl From<Neighbour> for u64 {
    fn from(neighbour: Neighbour) -> Self {
        let dist_bits = neighbour.dist.to_bits() as u64;
        let node_bits = neighbour.node as u64;
        (dist_bits << 32) | node_bits
    }
}

impl From<u64> for Neighbour {
    fn from(bits: u64) -> Self {
        let dist_bits = (bits >> 32) as u32;
        let node_bits = (bits & 0xFFFFFFFF) as u32;
        Neighbour {
            dist: f32::from_bits(dist_bits),
            node: node_bits,
        }
    }
}

enum DistMode<'a> {
    QKDist(&'a [f32]),
    KKDist
}

#[inline(always)]
fn prefetch_slice<T, const LOCALITY: i32, const LINES: i32>(slice: &[T]) {
    if slice.is_empty() || core::mem::size_of::<T>() == 0 { return; }
    const CL: usize = 64;
    let base = slice.as_ptr() as usize;
    let bytes = core::mem::size_of::<T>() * slice.len();
    let mut a = base & !(CL - 1);
    let end = min((base + bytes + CL - 1) & !(CL - 1), a + (LINES as usize) * CL);
    while a < end {
        prefetch_read_data::<_, LOCALITY>(a as *const u8);
        a += CL;
    }
}

impl<T: Sync + Clone + Value + Copy + InnerProduct + L2Square> Index<T> {
    #[inline(always)]
    fn distance(&self, a: ArrayView2<T>, b: &[T], mode: &DistMode) -> f32 {
        let r_sq = self.options.r_sq;

        // After raising dim, q'=(q; 0), k'=(k; sqrt(r^2 - |k|^2)). 
        return match mode {
            DistMode::QKDist(q_l2sq) => {
                // |q'-k'|^2 = |q|^2 + |k|^2 + (r^2 - |k|^2) - 2q.k = |q|^2 + r^2 - 2q.k
                f32::from_bits((0..a.shape()[0]).map(|i| {
                    let a = a.row(i);
                    let a = a.as_slice().unwrap();
                    (q_l2sq[i] + r_sq - 2.0 * T::dot(a, b)).to_bits()   // we can use bits for comparison since |q'-k'|^2 >= 0
                }).min().unwrap())
            },

            DistMode::KKDist => {
                // |k1'-k2'|^2 = |k1 - k2|^2 + (sqrt(r^2 - |k1|^2) - sqrt(r^2 - |k2|^2))^2
                assert!(a.shape()[0] == 1, "Group size greater than 1 not supported for KK distance");
                let a = a.row(0);
                let a = a.as_slice().unwrap();
                let (s1, s2) = (sqrtf32(r_sq - T::dot(a, a)), sqrtf32(r_sq - T::dot(b, b)));
                T::l2sq(a, b) + (s1 - s2) * (s1 - s2)
            }
        }
    }

    #[inline(always)]
    fn get_node(&self, id: NodeID) -> &Node<T> {
        &self.nodes[id as usize]
    }

    #[inline(always)]
    fn get_node_mut(&mut self, id: NodeID) -> &mut Node<T> {
        &mut self.nodes[id as usize]
    }

    #[inline(always)]
    fn get_offset(&self, id: NodeID) -> usize {
        id as usize * self.dim
    }

    #[inline(always)]
    fn get_key(&self, id: NodeID) -> &[T] {
        let start = self.get_offset(id);
        &self.keys[start..start + self.dim]
    }

    #[inline(always)]
    fn get_val(&self, id: NodeID) -> &[T] {
        let start = self.get_offset(id);
        &self.vals[start..start + self.dim]
    }

    // batched version of search_nn
    #[inline(always)]
    fn search_nn_batched(&self, q: [ArrayView2<T>; BATCH_SIZE], ep: [&[Neighbour]; BATCH_SIZE], ef: usize, dist_mode: &DistMode) -> [Vec<Neighbour>; BATCH_SIZE] {
        assert_eq!(q.len(), ep.len(), "Mismatch between query batch and entry points batch");
        let batch_size = q.len();
        assert!(batch_size <= BATCH_SIZE, "Batch size exceeds compile-time limit");

        if batch_size == 0 {
            return array::from_fn(|_| Vec::new());
        }

        let mut visited_handle = self.visited_set.get_handle();

        let mut candidates: [BinaryHeap<Reverse<u64>>; BATCH_SIZE] = array::from_fn(|_| BinaryHeap::new());
        let mut nns: [BinaryHeap<u64>; BATCH_SIZE] = array::from_fn(|_| BinaryHeap::new());
        let mut furthest = [Distance::INFINITY; BATCH_SIZE];
        let mut request_done = [true; BATCH_SIZE];
        let mut staging_nodes: [Vec<NodeID>; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        let mut staging_dists: [Vec<Distance>; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        let mut dot_buffers: [Vec<f32>; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        let mut q_row_counts = [0usize; BATCH_SIZE];

        // Pre-split q's L2 norms for DistMode::QKDist so we can reuse them during the search.
        let mut q_norm_slices: [Option<&[f32]>; BATCH_SIZE] = array::from_fn(|_| None);
        if let DistMode::QKDist(q_l2sq_all) = dist_mode {
            let mut offset = 0usize;
            for i in 0..batch_size {
                let rows = q[i].shape()[0];
                let end = offset + rows;
                assert!(end <= q_l2sq_all.len(), "q_l2sq provided to DistMode::QKDist is too small");
                q_norm_slices[i] = Some(&q_l2sq_all[offset..end]);
                offset = end;
            }
        }

        for i in 0..batch_size {
            let query = q[i];
            assert_eq!(query.shape()[1], self.dim, "Query dimension does not match index dimension");
            request_done[i] = false;

            q_row_counts[i] = query.shape()[0];

            let entry_points = ep[i];
            candidates[i].extend(entry_points.iter().map(|n| {
                let entry_bits: u64 = (*n).into();
                Reverse(entry_bits)
            }));
            nns[i].extend(entry_points.iter().map(|n| {
                let entry_bits: u64 = (*n).into();
                entry_bits
            }));
            nns[i].reserve(ef);

            for neighbour in entry_points.iter() {
                let pos = self.get_visited_pos(i, neighbour.node);
                visited_handle.insert(pos);
            }

            furthest[i] = nns[i].peek().map_or(Distance::INFINITY, |n| Neighbour::from(*n).dist);
        }

        let mut remaining = batch_size;
        let mut key_buffer: Vec<T> = Vec::new();

        while remaining > 0 {
            // Stage 1: expand neighbours for each active request to gather candidate nodes.
            let mut expanded_any = false;
            for i in 0..BATCH_SIZE {
                staging_nodes[i].clear();
                staging_dists[i].clear();

                if i >= batch_size || request_done[i] {
                    continue;
                }

                let next_candidate_bits = match candidates[i].peek() {
                    Some(Reverse(bits)) => *bits,
                    None => {
                        request_done[i] = true;
                        remaining -= 1;
                        continue;
                    }
                };

                let peek: Neighbour = next_candidate_bits.into();

                if peek.dist > furthest[i] && nns[i].len() >= ef {
                    request_done[i] = true;
                    remaining -= 1;
                    continue;
                }

                let Reverse(candidate_bits) = candidates[i].pop().unwrap();
                let candidate: Neighbour = candidate_bits.into();
                let node_ref = self.get_node(candidate.node);

                prefetch_read_data::<_, 3>(node_ref as *const _ as *const u8);

                let edges = &node_ref.edges[..node_ref.edges_len as usize];
                self.stats.num_expanded_nodes.fetch_add(1, Ordering::Relaxed);
                self.stats.num_candidates_raw.fetch_add(edges.len(), Ordering::Relaxed);

                let staging = &mut staging_nodes[i];
                staging.reserve(edges.len());

                for &node_id in edges.iter() {
                    prefetch_slice::<_, 2, 2>(self.get_key(node_id));
                    let visited_pos = self.get_visited_pos(i, node_id);
                    visited_handle.prefetch(visited_pos);
                    let visited = visited_handle.contains(visited_pos);
                    visited_handle.insert(visited_pos);
                    if !visited {
                        staging.push(node_id);
                    }
                }

                self.stats.num_candidates_unvisited.fetch_add(staging.len(), Ordering::Relaxed);
                expanded_any = true;
            }

            // If nothing expanded in this round, we can break out to avoid spinning.
            if !expanded_any {
                break;
            }

            // Stage 2: gather contiguous key vectors for all staging candidates across requests.
            let mut total_candidates = 0usize;
            let mut gather_offsets = [0usize; BATCH_SIZE];
            let mut gather_counts = [0usize; BATCH_SIZE];

            for i in 0..BATCH_SIZE {
                if i >= batch_size || request_done[i] {
                    continue;
                }
                let count = staging_nodes[i].len();
                gather_offsets[i] = total_candidates;
                gather_counts[i] = count;
                total_candidates += count;
            }

            if total_candidates == 0 {
                continue;
            }

            key_buffer.clear();
            key_buffer.reserve(total_candidates * self.dim);

            for i in 0..BATCH_SIZE {
                if i >= batch_size || request_done[i] {
                    continue;
                }
                for &node_id in staging_nodes[i].iter() {
                    let key_slice = self.get_key(node_id);
                    key_buffer.extend_from_slice(key_slice);
                }
            }

            // Stage 3: distance computation using batched matrix-style loops.
            match dist_mode {
                DistMode::QKDist(_) => {
                    for i in 0..BATCH_SIZE {
                        if i >= batch_size || request_done[i] {
                            continue;
                        }
                        let count = gather_counts[i];
                        if count == 0 {
                            continue;
                        }

                        let rows = q_row_counts[i];
                        if rows == 0 {
                            continue;
                        }

                        let norms = q_norm_slices[i].expect("Missing q L2 norms for DistMode::QKDist");
                        assert_eq!(norms.len(), rows, "q_l2sq slice length mismatch");

                        staging_dists[i].resize(count, 0.0);
                        dot_buffers[i].resize(rows * count, 0.0);

                        let base = gather_offsets[i] * self.dim;
                        let key_segment = &key_buffer[base..base + count * self.dim];

                        for row_idx in 0..rows {
                            let q_row = q[i].row(row_idx);
                            let q_slice = q_row.as_slice().expect("Query row is not contiguous");
                            let dst = &mut dot_buffers[i][row_idx * count..(row_idx + 1) * count];
                            for col in 0..count {
                                let key_slice = &key_segment[col * self.dim..(col + 1) * self.dim];
                                dst[col] = T::dot(q_slice, key_slice);
                            }
                        }

                        for col in 0..count {
                            let mut best_bits = Distance::INFINITY.to_bits();
                            for row_idx in 0..rows {
                                let dot = dot_buffers[i][row_idx * count + col];
                                let dist = norms[row_idx] + self.options.r_sq - 2.0 * dot;
                                let bits = dist.to_bits();
                                if bits < best_bits {
                                    best_bits = bits;
                                }
                            }
                            staging_dists[i][col] = f32::from_bits(best_bits);
                        }
                    }
                }
                DistMode::KKDist => {
                    for i in 0..BATCH_SIZE {
                        if i >= batch_size || request_done[i] {
                            continue;
                        }

                        let count = gather_counts[i];
                        if count == 0 {
                            continue;
                        }

                        let rows = q_row_counts[i];
                        assert!(rows == 1, "Group size greater than 1 not supported for KK distance");

                        let q_row = q[i].row(0);
                        let q_slice = q_row.as_slice().expect("Query row is not contiguous");
                        let s1 = sqrtf32(self.options.r_sq - T::dot(q_slice, q_slice));

                        staging_dists[i].resize(count, 0.0);
                        let base = gather_offsets[i] * self.dim;
                        let key_segment = &key_buffer[base..base + count * self.dim];

                        for col in 0..count {
                            let key_slice = &key_segment[col * self.dim..(col + 1) * self.dim];
                            let s2 = sqrtf32(self.options.r_sq - T::dot(key_slice, key_slice));
                            let l2 = T::l2sq(q_slice, key_slice);
                            staging_dists[i][col] = l2 + (s1 - s2) * (s1 - s2);
                        }
                    }
                }
            }

            // Stage 4: update candidate/result heaps with newly computed distances.
            for i in 0..BATCH_SIZE {
                if i >= batch_size || request_done[i] {
                    continue;
                }

                let staging = &staging_nodes[i];
                let dists = &staging_dists[i];
                assert_eq!(staging.len(), dists.len());

                for idx in 0..staging.len() {
                    let node_id = staging[idx];
                    let dist = dists[idx];

                    if dist >= furthest[i] && nns[i].len() >= ef {
                        self.stats.num_candidates_pruned.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    let entry = Neighbour { dist, node: node_id };
                    let entry_u64: u64 = entry.into();

                    candidates[i].push(Reverse(entry_u64));
                    if nns[i].len() >= ef {
                        *nns[i].peek_mut().unwrap() = entry_u64;
                    } else {
                        nns[i].push(entry_u64);
                    }

                    furthest[i] = nns[i].peek().map_or(Distance::INFINITY, |n| Neighbour::from(*n).dist);
                }
            }
        }

        // Finalise results per request.
        let mut results: [Vec<Neighbour>; BATCH_SIZE] = array::from_fn(|_| Vec::new());
        for i in 0..batch_size {
            let neighbours = nns[i]
                .clone()
                .into_sorted_vec()
                .into_iter()
                .map(|bits| Neighbour::from(bits))
                .collect::<Vec<_>>();
            results[i] = neighbours;
        }

        results
    }

    /// Searches for `ef` nearest neighbors of a query vector `q` in the layer from entry points `ep`.
    /// Note that the returned neighbors are sorted by distance (from min to max)
    #[inline(always)]
    fn search_nn(&self, q: ArrayView2<T>, ep: &[Neighbour], ef: usize, dist_mode: &DistMode) -> Vec<Neighbour> {
        let mut candidates: BinaryHeap<Reverse<u64>> = ep.iter().map(|n| Reverse(n.clone().into())).collect(); // candidate nearest neighbors (min-heap)
        let mut nns       : BinaryHeap<u64>          = ep.iter().map(|n| n.clone().into()).collect();          // result nearest neighbors    (max-heap)

        let mut furthest = nns.peek().map_or(Distance::INFINITY, |n| Neighbour::from(*n).dist);

        let mut visited_set = self.visited_set.get_handle();
        ep.iter().map(|n| n.node).for_each(|id| {
            visited_set.insert(id as usize);
        });

        nns.reserve(ef);

        // BFS from entry points
        while let Some(Reverse(peek)) = candidates.peek() {
            let peek: Neighbour = (*peek).into();
            let candidate = self.get_node(peek.node);

            prefetch_read_data::<_, 3>(candidate as *const _ as *const u8);

            // early stopping condition
            if peek.dist > furthest && nns.len() >= ef {
                break;
            }

            // pop the nearest candidate
            candidates.pop();
            self.stats.num_expanded_nodes.fetch_add(1, Ordering::Relaxed);

            // gather unvisited neighbors of the candidate
            let edges = &candidate.edges[..candidate.edges_len as usize];
            self.stats.num_candidates_raw.fetch_add(edges.len(), Ordering::Relaxed);
            let mut staging = vec![(0u32, 0.0f32); edges.len()];
            let mut staging_len = 0;
            for i in 0..edges.len() {
                let node_id = edges[i];

                prefetch_slice::<_, 2, 2>(self.get_key(node_id));

                // skip if already visited
                let visited = visited_set.contains(node_id as usize);
                visited_set.insert(node_id as usize);
                staging[staging_len] = (node_id, 0.0);
                if !visited {
                    staging_len += 1;
                }
            }
            staging.truncate(staging_len);
            self.stats.num_candidates_unvisited.fetch_add(staging.len(), Ordering::Relaxed);

            // distance calculation
            for i in 0..staging.len() {
                staging[i].1 = self.distance(q, self.get_key(staging[i].0), &dist_mode);
            }

            // update candidates and nns
            for &node in staging.iter() {
                let node_id = node.0;
                let dist = node.1;

                if dist >= furthest && nns.len() >= ef {
                    // skip if the distance is greater than the furthest in the result set
                    self.stats.num_candidates_pruned.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // add to candidates
                candidates.push(Reverse(Neighbour { dist, node: node_id }.into()));
                let neigh = Neighbour { dist, node: node_id };
                if nns.len() >= ef {
                    // if the result set exceeds ef, remove the furthest neighbor
                    *nns.peek_mut().unwrap() = neigh.into();
                } else {
                    nns.push(neigh.into());
                }
                furthest = nns.peek().map_or(Distance::INFINITY, |n| Neighbour::from(*n).dist);
            }
        }

        nns.into_sorted_vec().into_iter().map(|n| Neighbour::from(n)).collect()
    }

    #[inline(always)]
    fn get_visited_pos(&self, batch_id: usize, node_id: NodeID) -> usize {
        batch_id * self.options.n_max + node_id as usize
    }

    /// Select `q`'s `m` neighbors from `candidates`.
    /// The selected neighbors must not be too close to each other.
    #[inline(always)]
    fn select_neighbors(&self, candidates: &[Neighbour], m: usize) -> Vec<Neighbour> {
        if candidates.len() <= m {
            return candidates.iter().cloned().collect();
        }

        let mut candidates: BinaryHeap<Reverse<u64>> = candidates.iter().map(|n| Reverse(n.clone().into())).collect();
        let mut selected: Vec<Neighbour> = vec![];

        'select: while !candidates.is_empty() && selected.len() < m {
            let candidate: Neighbour = candidates.pop().unwrap().0.into();

            for sel in selected.iter() {
                let a = self.get_key(sel.node);
                let a = ArrayView2::from_shape((1, a.len()), a).unwrap();
                let dist = self.distance(a, self.get_key(candidate.node), &DistMode::KKDist);
                if dist < candidate.dist {
                    // if we select the candidate, the distance between candidate and selected node is too small
                    // skip this candidate
                    continue 'select;
                }
            }

            // if we reach here, the candidate is valid
            selected.push(candidate.clone());
        }

        selected
    }

    fn add_conn<'g>(&mut self, from: NodeID, to: NodeID) {
        let m_max = self.m_max;
        let node = self.get_node(from);
        let node_vec = self.get_key(from);

        let mut edges = node.edges[..node.edges_len as usize].to_vec();
        assert!(!edges.contains(&to), "Connection already exists from {:?} to {:?}", from, to);

        // add new connection
        edges.push(to);

        // shrink if necessary
        if edges.len() > m_max {
            let candidates = edges.iter().map(|&dst_id| {
                let a = ArrayView2::from_shape((1, node_vec.len()), node_vec).unwrap();
                Neighbour { dist: self.distance(a, self.get_key(dst_id), &DistMode::KKDist), node: dst_id }
            }).collect::<Vec<_>>();
            edges = self.select_neighbors(&candidates, m_max).iter().map(|n| n.node).collect();
        }

        let node = self.get_node_mut(from);
        node.edges = [NodeID::MAX; MAX_NUM_NEIGHBOURS];
        node.edges[..edges.len()].copy_from_slice(&edges);
        node.edges_len = edges.len() as u8;
    }

    pub fn clear_stats(&mut self) {
        self.stats = Statistic::new();
    }

    pub fn insert(&mut self, node_id: NodeID, key: ArrayView1<T>, val: ArrayView1<T>, ep: ArrayView1<Neighbour>) -> Result<(), ()> {
        assert_eq!(key.shape(), [self.dim], "Vector dimension does not match index dimension");
        let key = key.as_slice().unwrap();
        let val = val.as_slice().unwrap();
        let ep = ep.as_slice().unwrap();

        // write vector
        let vec_start = self.get_offset(node_id);
        self.keys[vec_start..vec_start + self.dim].copy_from_slice(&key);
        self.vals[vec_start..vec_start + self.dim].copy_from_slice(&val);

        // search nearest neighbours
        if !ep.is_empty() {
            // expand and select neighbours
            let key = ArrayView2::from_shape((1, self.dim), key).unwrap();
            let ep = self.search_nn(key, ep, self.options.ef_cons, &DistMode::KKDist);
            let neighbours = self.select_neighbors(&ep, self.options.m).iter().map(|n| n.node).collect::<Vec<_>>();

            // create node and node -> neighbor connection
            let mut edges = [NodeID::MAX; MAX_NUM_NEIGHBOURS];
            edges[..neighbours.len()].copy_from_slice(&neighbours);
            *self.get_node_mut(node_id) = Node {
                valid: true,
                edges,
                edges_len: neighbours.len() as u8,
                phantom: PhantomData,
            };

            // create reverse connection neighbor -> node
            for &neighbor in neighbours.iter() {
                self.add_conn(neighbor, node_id);
            }
        } else {
            // create empty node
            *self.get_node_mut(node_id) = Node {
                valid: true,
                edges: [NodeID::MAX; MAX_NUM_NEIGHBOURS],
                edges_len: 0,
                phantom: PhantomData,
            };
        }

        Ok(())
    }

    pub fn search_batched(&self, query: [ArrayView2<T>; BATCH_SIZE], ef: usize, ep: [&[Neighbour]; BATCH_SIZE]) -> [Vec<Neighbour>; BATCH_SIZE] {
        for i in 0..query.len() {
            assert_eq!(query[i].shape()[1], self.dim, "Query dimension does not match index dimension");
        }

        self.stats.num_searches.fetch_add(query.len(), Ordering::Relaxed);

        let start = Instant::now();

        // calculate q_l2sq
        let mut q_l2sq_all: Vec<f32> = vec![];
        for i in 0..query.len() {
            let q_l2sq = query[i].rows().into_iter().map(|row| {
                let row = row.as_slice().unwrap();
                T::dot(row, row)
            }).collect::<Vec<_>>();
            q_l2sq_all.extend_from_slice(&q_l2sq);
        }

        // search at level 0
        let neighbors = self.search_nn_batched(query, ep, ef, &DistMode::QKDist(&q_l2sq_all));

        let search_time = start.elapsed().as_micros() as usize;
        self.stats.search_latency.fetch_add(search_time, Ordering::Relaxed);

        neighbors        
    }

    // return (ef, dim) of (K, V)
    pub fn search(&self, query: ArrayView2<T>, ef: usize, ep: ArrayView1<Neighbour>) -> (Array2<T>, Array2<T>) {
        assert_eq!(query.shape()[1], self.dim, "Query dimension does not match index dimension");
        let ep = ep.as_slice().unwrap();

        self.stats.num_searches.fetch_add(1, Ordering::Relaxed);

        let start = Instant::now();

        // calculate q_l2sq
        let q_l2sq = query.rows().into_iter().map(|row| {
            let row = row.as_slice().unwrap();
            T::dot(row, row)
        }).collect::<Vec<_>>();

        // search at level 0
        let neighbors = self.search_nn(query, &ep, ef, &DistMode::QKDist(&q_l2sq));

        let search_time = start.elapsed().as_micros() as usize;
        self.stats.search_latency.fetch_add(search_time, Ordering::Relaxed);

        // collect
        assert!(neighbors.len() <= ef, "Number of neighbors returned exceeds ef");
        let mut keys = Array2::<T>::from_elem((ef, self.dim), T::zero());
        let mut vals = Array2::<T>::from_elem((ef, self.dim), T::zero());
        for i in 0..neighbors.len() {
            let neighbor = &neighbors[i];
            let key_slice = self.get_key(neighbor.node);
            let val_slice = self.get_val(neighbor.node);
            keys.slice_mut(s![i, ..key_slice.len()]).assign(&ArrayView1::from(key_slice));
            vals.slice_mut(s![i, ..val_slice.len()]).assign(&ArrayView1::from(val_slice));
        }
        (keys, vals)
    }

    pub fn new(dim: usize, options: Options) -> Self {
        let m_max = options.m * 2;
        assert!(m_max <= MAX_NUM_NEIGHBOURS, "Maximum number of neighbors at layer 0 exceeded");
        Index {
            nodes: repeat_with(|| Node {
                valid: false,
                edges: [NodeID::MAX; MAX_NUM_NEIGHBOURS],
                edges_len: 0,
                phantom: PhantomData,
            }).take(options.n_max).collect::<Vec<_>>().into(),
            keys: repeat_with(|| T::zero()).take(options.n_max * dim).collect::<Vec<_>>().into(),
            vals: repeat_with(|| T::zero()).take(options.n_max * dim).collect::<Vec<_>>().into(),
            m_max,
            dim,
            visited_set: TLSet::new(options.n_max * BATCH_SIZE),
            stats: Statistic::new(),
            options,
        }
    }

    fn take_snapshot(&self) -> IndexSnapshot {
        let mut nodes: Vec<NodeSnapshot> = vec![];
        for id in 0..self.options.n_max {
            let node = self.get_node(id as NodeID);
            if !node.valid {
                break;
            }
            let edges = node.edges[..node.edges_len as usize].to_vec();
            nodes.push(NodeSnapshot {
                id: id as NodeID,
                edges
            });
        }

        let mut keys = vec![];
        let mut vals = vec![];
        for id in 0..nodes.len() {
            keys.extend(self.get_key(id as NodeID).into_iter().map(|v| v.to_f32()));
            vals.extend(self.get_val(id as NodeID).into_iter().map(|v| v.to_f32()));
        }

        IndexSnapshot {
            nodes,
            keys,
            vals,
            options: self.options.clone(),
            dim: self.dim,
        }
    }

    fn from_snapshot(snapshot: IndexSnapshot) -> Self {
        let options = snapshot.options;
        let dim = snapshot.dim;

        let nodes: Vec<Node<T>> = snapshot.nodes.into_iter().map(|node| {
            let mut edges = [NodeID::MAX; MAX_NUM_NEIGHBOURS];
            edges[..node.edges.len()].copy_from_slice(&node.edges);
            Node {
                valid: true,
                edges,
                edges_len: node.edges.len() as u8,
                phantom: PhantomData,
            }
        }).collect();

        let src = snapshot.keys.into_iter().map(|v| T::from_f32(v)).collect::<Vec<_>>();
        let mut keys: Vec<T> = repeat_with(|| T::zero()).take(nodes.len() * dim).collect();
        keys[..src.len()].copy_from_slice(&src);

        let src = snapshot.vals.into_iter().map(|v| T::from_f32(v)).collect::<Vec<_>>();
        let mut vals: Vec<T> = repeat_with(|| T::zero()).take(nodes.len() * dim).collect();
        vals[..src.len()].copy_from_slice(&src);

        let m_max = options.m * 2;
        assert!(m_max <= MAX_NUM_NEIGHBOURS, "Maximum number of neighbors at layer 0 exceeded");

        Index {
            nodes: nodes.into(),
            keys: keys.into(),
            vals: vals.into(),
            m_max,
            dim,
            visited_set: TLSet::new(options.n_max * BATCH_SIZE),
            stats: Statistic::new(),
            options,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        let snapshot = self.take_snapshot();
        bincode::serde::encode_into_std_write(&snapshot, &mut writer, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let snapshot: IndexSnapshot = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Self::from_snapshot(snapshot))
    }
}
