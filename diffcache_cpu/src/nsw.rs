use crate::{Distance, HugeArray, L2Square, Neighbour, NodeID};
use std::{array, cmp::{Reverse, max, min}, collections::{BinaryHeap, HashSet}, f32, intrinsics::{prefetch_read_data, sqrtf32}, io::{BufReader, BufWriter}, iter::repeat_with, marker::PhantomData, ops::{Add, Div}, path::Path, sync::atomic::Ordering, time::Instant};
use crate::{Value};
use std::cell::RefCell;
use super::tlset::TLSet;
use metrics::counter;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, s};
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
    pub name: String,
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

pub struct Index<T: Clone> {
    nodes: HugeArray<Node<T>>,
    keys: HugeArray<T>,
    vals: HugeArray<T>,
    options: Options,
    m_max: usize,
    dim: usize,
    visited_set: TLSet,
    num_nodes: usize
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
    num_nodes: usize,
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

enum DistMode {
    QKDist,
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
    // NOTE: distance should be positive (will be compared using bits)
    #[inline(always)]
    fn distance(&self, a: &[T], b: &[T], mode: &DistMode) -> f32 {
        let r_sq = self.options.r_sq;

        // After raising dim, q'=(q; 0), k'=(k; sqrt(r^2 - |k|^2)). 
        return match mode {
            DistMode::QKDist => {
                // mean(|q'-k'|^2) = mean(|q|^2 + |k|^2 + (r^2 - |k|^2) - 2q.k) = mean(|q|^2 + r^2) - 2q.k = constant - 2q.k
                // FIXME: we assume that 65536.0 - T::dot(a, b) >= 0
                65536.0 - T::dot(a, b)
            },

            DistMode::KKDist => {
                // |k1'-k2'|^2 = |k1 - k2|^2 + (sqrt(r^2 - |k1|^2) - sqrt(r^2 - |k2|^2))^2
                //             = 2r^2 - 2k1.k2 - 2sqrt(r^2 - |k1|^2)sqrt(r^2 - |k2|^2)
                let (s1, s2) = (sqrtf32(f32::max(r_sq - T::dot(a, a), 0.0)), sqrtf32(f32::max(r_sq - T::dot(b, b), 0.0)));
                r_sq - T::dot(a, b) - s1 * s2
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
    fn get_print_key(&self, id: NodeID) -> Vec<f32> {
        let start = self.get_offset(id);
        let mut key: Vec<f32> = self.keys[start..start + self.dim].iter().map(|v| v.to_f32()).collect();
        key.push(sqrtf32(self.options.r_sq - T::dot(&self.keys[start..start + self.dim], &self.keys[start..start + self.dim])));
        key
    }

    #[inline(always)]
    fn get_print_query(&self, q: ArrayView2<T>) -> Vec<Vec<f32>> {
        let mut res: Vec<Vec<f32>> = vec![];
        for i in 0..q.shape()[0] {
            let row = q.row(i);
            let row = row.as_slice().unwrap();
            let mut key: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
            key.push(0.0f32);
            res.push(key);
        }
        res
    }

    #[inline(always)]
    fn get_val(&self, id: NodeID) -> &[T] {
        let start = self.get_offset(id);
        &self.vals[start..start + self.dim]
    }

    /// Searches for `ef` nearest neighbors of a query vector `q` in the layer from entry points `ep`.
    /// Note that the returned neighbors are sorted by distance (from min to max)
    #[inline(always)]
    fn search_nn(&self, q: &[T], ep: &[Neighbour], ef: usize, dist_mode: DistMode) -> Vec<Neighbour> {
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
            counter!(format!("{}_num_expanded_nodes", self.options.name)).increment(1);

            // gather unvisited neighbors of the candidate
            let edges = &candidate.edges[..candidate.edges_len as usize];
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

            // distance calculation
            counter!(format!("{}_num_accessed_nodes", self.options.name)).increment(staging.len() as u64);
            for i in 0..staging.len() {
                staging[i].1 = self.distance(q, self.get_key(staging[i].0), &dist_mode);
            }

            // update candidates and nns
            for &node in staging.iter() {
                let node_id = node.0;
                let dist = node.1;

                if dist >= furthest && nns.len() >= ef {
                    // skip if the distance is greater than the furthest in the result set
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
                let dist = self.distance(self.get_key(sel.node), self.get_key(candidate.node), &DistMode::KKDist);
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
                Neighbour { dist: self.distance(node_vec, self.get_key(dst_id), &DistMode::KKDist), node: dst_id }
            }).collect::<Vec<_>>();
            edges = self.select_neighbors(&candidates, m_max).iter().map(|n| n.node).collect();
        }

        let node = self.get_node_mut(from);
        node.edges = [NodeID::MAX; MAX_NUM_NEIGHBOURS];
        node.edges[..edges.len()].copy_from_slice(&edges);
        node.edges_len = edges.len() as u8;
    }

    pub fn prefill(&mut self, keys: ArrayView2<T>, vals: ArrayView2<T>, neighbours: ArrayView2<NodeID>) -> Result<(), ()> {
        assert_eq!(keys.shape()[0], vals.shape()[0], "Number of keys and values must be the same");
        assert_eq!(keys.shape()[1], self.dim, "Vector dimension does not match index dimension");
        assert_eq!(vals.shape()[1], self.dim, "Vector dimension does not match index dimension");
        assert_eq!(neighbours.shape()[0], keys.shape()[0], "Number of keys and neighbours must be the same");

        let n = keys.shape()[0];
        let deg = neighbours.shape()[1];

        // insert keys and vals
        self.keys[..n * self.dim].copy_from_slice(keys.as_slice().unwrap());
        self.vals[..n * self.dim].copy_from_slice(vals.as_slice().unwrap());

        // build conns
        assert!(deg <= self.m_max, "Number of neighbors exceeds maximum allowed");
        for i in 0..n {
            let node_id = i as NodeID;
            *self.get_node_mut(node_id) = Node {
                valid: true,
                edges: {
                    let mut edges = [NodeID::MAX; MAX_NUM_NEIGHBOURS];
                    edges[..deg].copy_from_slice(&neighbours.row(i).as_slice().unwrap());
                    edges
                },
                edges_len: deg as u8,
                phantom: PhantomData,
            };
        }

        Ok(())
    }

    fn get_ep<'a>(&self, key: &[T], ep: &[Neighbour], dist_mode: &DistMode) -> Option<Vec<Neighbour>> {
        if !ep.is_empty() {
            Some(ep.to_vec())
        } else if self.num_nodes == 0 {
            // empty index
            None
        } else {
            // ramdom entry point
            let node_id = (self.num_nodes as f32 * rand::random::<f32>()) as NodeID;
            assert!(node_id < self.num_nodes as NodeID, "Random entry point node ID out of range");
            let dist = self.distance(key, self.get_key(node_id), dist_mode);
            Some(vec![Neighbour { dist, node: node_id }])
        }
    }

    pub fn insert(&mut self, key: ArrayView1<T>, val: ArrayView1<T>, ep: ArrayView1<Neighbour>) -> Result<(), ()> {
        assert_eq!(key.shape(), [self.dim], "Vector dimension does not match index dimension");
        let key = key.as_slice().unwrap();
        let val = val.as_slice().unwrap();
        let ep = self.get_ep(key, ep.as_slice().unwrap(), &DistMode::KKDist);

        // get node ID
        let node_id = self.num_nodes as NodeID;
        self.num_nodes += 1;

        // write vector
        let vec_start = self.get_offset(node_id);
        self.keys[vec_start..vec_start + self.dim].copy_from_slice(&key);
        self.vals[vec_start..vec_start + self.dim].copy_from_slice(&val);

        // search nearest neighbours
        if let Some(ep) = ep {
            // expand and select neighbours
            let ep = self.search_nn(key, &ep, self.options.ef_cons, DistMode::KKDist);
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

    pub fn search_bruteforce(&self, query: ArrayView1<T>, ef: usize, mut out_keys: ArrayViewMut2<T>, mut out_vals: ArrayViewMut2<T>) {
        assert_eq!(query.shape()[0], self.dim, "Query dimension does not match index dimension");

        counter!(format!("{}_num_searches", self.options.name)).increment(1);

        let start = Instant::now();

        // brute-force search
        let mut heap: BinaryHeap<u64> = BinaryHeap::with_capacity(ef); // max-heap
        for node_id in 0..self.options.n_max as NodeID {
            let node = self.get_node(node_id);
            if !node.valid {
                break;
            }
            let key = self.get_key(node_id);
            let dist = self.distance(query.as_slice().unwrap(), key, &DistMode::QKDist);

            let neigh = Neighbour { dist, node: node_id };
            if heap.len() < ef {
                heap.push(neigh.clone().into());
            } else if dist < Neighbour::from(*heap.peek().unwrap()).dist {
                *heap.peek_mut().unwrap() = neigh.clone().into();
            }
        }

        let search_time = start.elapsed().as_micros() as usize;
        counter!(format!("{}_search_latency", self.options.name)).increment(search_time as u64);

        // collect
        let neighbors = heap.into_sorted_vec().into_iter().map(|n| Neighbour::from(n)).collect::<Vec<_>>();
        assert!(neighbors.len() <= ef, "Number of neighbors returned exceeds ef");
        for i in 0..neighbors.len() {
            let neighbor = &neighbors[i];
            let key_slice = self.get_key(neighbor.node);
            let val_slice = self.get_val(neighbor.node);
            out_keys.slice_mut(s![i, ..key_slice.len()]).assign(&ArrayView1::from(key_slice));
            out_vals.slice_mut(s![i, ..val_slice.len()]).assign(&ArrayView1::from(val_slice));
        }
    }

    // return (ef, dim) of (K, V)
    pub fn search(&self, query: ArrayView1<T>, ef: usize, ep: ArrayView1<Neighbour>, mut out_keys: ArrayViewMut2<T>, mut out_vals: ArrayViewMut2<T>) {
        assert_eq!(query.shape()[0], self.dim, "Query dimension does not match index dimension");
        let query = query.as_slice().unwrap();
        let ep = self.get_ep(query, ep.as_slice().unwrap(), &DistMode::QKDist).unwrap();

        counter!(format!("{}_num_searches", self.options.name)).increment(1);

        let start = Instant::now();

        // search at level 0
        let neighbors = self.search_nn(query, &ep, ef, DistMode::QKDist);

        let search_time = start.elapsed().as_micros() as usize;
        counter!(format!("{}_search_latency", self.options.name)).increment(search_time as u64);

        // collect
        assert!(neighbors.len() <= ef, "Number of neighbors returned exceeds ef");
        for i in 0..neighbors.len() {
            let neighbor = &neighbors[i];
            let key_slice = self.get_key(neighbor.node);
            let val_slice = self.get_val(neighbor.node);
            out_keys.slice_mut(s![i, ..key_slice.len()]).assign(&ArrayView1::from(key_slice));
            out_vals.slice_mut(s![i, ..val_slice.len()]).assign(&ArrayView1::from(val_slice));
        }
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
            options,
            num_nodes: 0,
        }
    }

    pub fn update_r_sq(&mut self, r_sq: f32) {
        self.options.r_sq = r_sq;
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
            num_nodes: self.num_nodes,
        }
    }

    fn from_snapshot(snapshot: IndexSnapshot) -> Self {
        let options = snapshot.options;
        let dim = snapshot.dim;

        let mut nodes: Vec<Node<T>> = snapshot.nodes.into_iter().map(|node| {
            let mut edges = [NodeID::MAX; MAX_NUM_NEIGHBOURS];
            edges[..node.edges.len()].copy_from_slice(&node.edges);
            Node {
                valid: true,
                edges,
                edges_len: node.edges.len() as u8,
                phantom: PhantomData,
            }
        }).collect();
        nodes.resize_with(options.n_max, || Node {
            valid: false,
            edges: [NodeID::MAX; MAX_NUM_NEIGHBOURS],
            edges_len: 0,
            phantom: PhantomData,
        });

        let src = snapshot.keys.into_iter().map(|v| T::from_f32(v)).collect::<Vec<_>>();
        let mut keys: Vec<T> = repeat_with(|| T::zero()).take(options.n_max * dim).collect();
        keys[..src.len()].copy_from_slice(&src);

        let src = snapshot.vals.into_iter().map(|v| T::from_f32(v)).collect::<Vec<_>>();
        let mut vals: Vec<T> = repeat_with(|| T::zero()).take(options.n_max * dim).collect();
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
            options,
            num_nodes: snapshot.num_nodes,
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

    pub fn get_keys(&self, len: usize) -> ArrayView2<'_, T> {
        ArrayView2::from_shape((len, self.dim), &self.keys[..len * self.dim]).unwrap()
    }
}
