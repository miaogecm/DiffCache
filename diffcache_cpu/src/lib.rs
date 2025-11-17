//! # DiffCache: Differential KVCache Indexing for Fast, Memory-efficient LLM Inference
//! 
//! This library contains the on-CPU data layer of DiffCache. In diff cache, on-GPU index
//! layer and on-CPU data layer work cooperatively to index important tokens in the KVCache.
//! The index layer search for high-quality entry points (eps) for the data layer, and the 
//! data layer retrieves relevant KV pairs based on these entry points.
//! 
//! ## Core Interfaces
//! 
//! (0) new(bsz, qkv_mapping, kv_head_num, head_dim)
//!     Create a DiffCache data layer instance
//! 
//! (1) insert(self, K, V, ep)                   (KV: (bsz,     kv_head_num, head_dim))
//!                                              (ep: (bsz,     kv_head_num, num_seeds))
//!     Insert KV into the cache                 
//! 
//! (2) query(self, Q, ef, ep)                   (Q:  (bsz,     q_head_num , head_dim))
//!                                              (ep: (bsz,     kv_head_num, num_seeds))
//!                                 (ret: (K, V) with (bsz, ef, kv_head_num, head_dim))
//!     Query the cache with Q to retrieve top ef relevant KVs
//! 
#![feature(portable_simd, pointer_is_aligned_to, core_intrinsics, stmt_expr_attributes, iterator_try_collect)]

use std::{ops::{Add, Deref, DerefMut, Div}, sync::{Arc, atomic::{AtomicBool, AtomicUsize}}};
use ndarray::{Array, Array2, Array4, ArrayView, ArrayView2, ArrayView3, ArrayView4, Data, Dimension, ShapeBuilder, s};
use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray3, PyReadonlyArray4};
use rand::rand_core::le;
use simsimd::{bf16, SpatialSimilarity};
use std::sync::RwLock;
use threadpool::{ThreadPool, Builder};
use crossbeam::channel;
use pyo3::{prelude::*, types::PyDict};
use metrics_exporter_tcp::TcpBuilder;
use std::cell::Cell;
use core_affinity;
use std::sync::LazyLock;

mod tlset;
mod nsw;

pub type Distance = f32;

type NodeID = u32;

thread_local! {
    static AFFINITY_SET: Cell<bool> = Cell::new(false);
}

static NEXT_CORE: AtomicUsize = AtomicUsize::new(0);

static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| Builder::new()
                                                            .num_threads(4)
                                                            .build());

fn try_bind_core() {
    AFFINITY_SET.with(|affinity_set| {
        if !affinity_set.get() {
            let cores = core_affinity::get_core_ids().unwrap();
            let core_id = NEXT_CORE.fetch_add(1, std::sync::atomic::Ordering::SeqCst) % cores.len();
            core_affinity::set_for_current(cores[core_id]);
            println!("Binding thread {:?} to core {:?}", std::thread::current().id(), cores[core_id]);
            affinity_set.set(true);
        }
    });
}

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub trait InnerProduct {
    fn dot(a: &[Self], b: &[Self]) -> f32 where Self: Sized;
}

pub trait L2Square {
    fn l2sq(a: &[Self], b: &[Self]) -> f32 where Self: Sized;
}

pub trait Mean {
    fn mean(vecs: &[Self], out: &mut [Self]) where Self: Sized;
}

impl InnerProduct for bf16 {
    fn dot(a: &[Self], b: &[Self]) -> f32 where Self: Sized {
        SpatialSimilarity::dot(a, b).unwrap() as f32
    }
}

impl L2Square for bf16 {
    fn l2sq(a: &[Self], b: &[Self]) -> f32 where Self: Sized {
        SpatialSimilarity::l2sq(a, b).unwrap() as f32
    }
}

impl Mean for bf16 {
    fn mean(vecs: &[Self], out: &mut [Self]) where Self: Sized {
        let dim = out.len();
        let n = vecs.len() / dim;
        let acc = vecs.chunks(dim).fold(vec![0.0f32; dim], |mut acc, v| {
            for i in 0..dim {
                acc[i] += bf16::to_f32(v[i]);
            }
            acc
        });
        for i in 0..dim {
            out[i] = bf16::from_f32(acc[i] / n as f32);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Neighbour {
    dist: Distance,
    node: NodeID,
}

pub(crate) struct HugeArray<T> {
    data: *mut T,
    size: usize,
}

unsafe impl<T: Send> Send for HugeArray<T> {}
unsafe impl<T: Sync> Sync for HugeArray<T> {}

impl <T: Copy> HugeArray<T> {
    pub(crate) fn new_with(size: usize, value: T) -> Self {
        let layout = std::alloc::Layout::array::<T>(size).unwrap();
        let ptr = hugepage_rs::alloc(layout) as *mut T;
        unsafe {
            for i in 0..size {
                ptr.add(i).write(value);
            }
        }
        HugeArray {
            data: ptr,
            size,
        }
    }
}

impl<T> Drop for HugeArray<T> {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::array::<T>(self.size).unwrap();
        hugepage_rs::dealloc(self.data as *mut u8, layout);
    }
}

impl<T> Deref for HugeArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe {
            std::slice::from_raw_parts(self.data, self.size)
        }
    }
}

impl<T> DerefMut for HugeArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            std::slice::from_raw_parts_mut(self.data, self.size)
        }
    }
}

impl<T> From<Vec<T>> for HugeArray<T> {
    fn from(vec: Vec<T>) -> Self {
        let mut vec = vec;
        let len = vec.len();
        let layout = std::alloc::Layout::array::<T>(len).unwrap();
        let ptr = hugepage_rs::alloc(layout) as *mut T;
        let src = vec.as_ptr();
        unsafe {
            vec.set_len(0);
            for i in 0..len {
                ptr.add(i).write(src.add(i).read());
            }
        }
        HugeArray {
            data: ptr,
            size: len,
        }
    }
}

pub trait Value {
    fn zero() -> Self;
    fn is_zero(&self) -> bool;
    fn from_f32(v: f32) -> Self;
    fn to_f32(&self) -> f32;
}

impl Value for bf16 {
    fn zero() -> Self {
        bf16::from_f32(0.0)
    }

    fn is_zero(&self) -> bool {
        self.to_f32() == 0.0
    }

    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }

    fn to_f32(&self) -> f32 {
        bf16::to_f32(*self)
    }
}

pub struct Options {
    max_ctx_len: usize,
    m: usize,
    ef_cons: usize,
    r_sq: Vec<f32>,
    batch_search: bool,
    name: String,
    use_bruteforce: bool
}

pub struct DataLayer<T: Clone> {
    indexes: Vec<Vec<Arc<RwLock<nsw::Index<T>>>>>,       // per batch per KV head index
    ctx_len: usize,                                      // current context length
    bsz: usize,                                          // batch size
    kv_head_num: usize,                                  // number of KV heads
    q_head_num: usize,                                   // number of Q heads
    head_dim: usize,                                     // dimension per head
    opt: Options,
}

pub struct Handle<R> {
    result: R,
}

impl<R> Handle<R> {
    pub fn completed(&self) -> bool {
        THREAD_POOL.queued_count() == 0 && THREAD_POOL.active_count() == 0
    }

    pub fn result(self) -> R {
        THREAD_POOL.join();
        self.result
    }
}

pub struct InsertResult;

pub struct QueryResult<T: Clone + Copy> {
    rx: channel::Receiver<(usize, usize, Array2<T>, Array2<T>)>,  // (bsz, kv_head_num or query_head_num, key=(ef, head_dim), val=(ef, head_dim))
    bsz: usize,
    head_num: usize,
    ef: usize,
    head_dim: usize,
}

impl<T: Clone + Copy + Value> QueryResult<T> {
    pub fn collect(self) -> (Array4<T>, Array4<T>) {
        let shape = (self.bsz, self.ef, self.head_num, self.head_dim);
        let mut keys = Array4::<T>::from_elem(shape, T::zero());
        let mut vals = Array4::<T>::from_elem(shape, T::zero());

        for _ in 0..self.bsz*self.head_num {
            let (b, kh, k_ret, v_ret) = self.rx.recv().unwrap();
            assert!(k_ret.shape() == &[self.ef, self.head_dim]);
            assert!(v_ret.shape() == &[self.ef, self.head_dim]);
            keys.slice_mut(s![b, .., kh, ..]).assign(&k_ret);
            vals.slice_mut(s![b, .., kh, ..]).assign(&v_ret);
        }

        assert!(self.rx.try_recv().is_err(), "Extra results received in QueryResult");

        (keys, vals)
    }
}

impl<T: 'static + Send + Sync + Clone + Value + Copy + InnerProduct + L2Square + Mean> DataLayer<T> {
    pub fn new(bsz: usize, q_head_num: usize, kv_head_num: usize, head_dim: usize, opt: Options) -> Self {
        assert!(q_head_num % kv_head_num == 0, "q_head_num must be multiple of kv_head_num");

        let mut indexes = vec![];

        for _ in 0..bsz {
            let mut batch_indexes = vec![];

            for h in 0..kv_head_num {
                batch_indexes.push(Arc::new(RwLock::new(nsw::Index::new(
                    head_dim,
                    nsw::Options {
                        n_max: opt.max_ctx_len,
                        m: opt.m,
                        ef_cons: opt.ef_cons,
                        r_sq: opt.r_sq[h],
                        name: format!("{}b{:03}h{:03}", opt.name, indexes.len(), h),
                    }
                ))));
            }

            indexes.push(batch_indexes);
        }

        DataLayer {
            indexes,
            ctx_len: 0,
            bsz,
            kv_head_num,
            q_head_num,
            head_dim,
            opt
        }
    }

    // keys: (bsz, seq_len, kv_head_num, head_dim)
    // vals: (bsz, seq_len, kv_head_num, head_dim)
    // neighbours: (bsz, num_kv_heads, seq_len, deg) u32
    pub fn prefill(&mut self, keys: ArrayView4<T>, vals: ArrayView4<T>, neighbours: ArrayView4<u32>) -> Handle<InsertResult> {
        let seq_len = keys.shape()[1];

        assert_eq!(keys.shape(), &[self.bsz, seq_len, self.kv_head_num, self.head_dim]);
        assert_eq!(vals.shape(), &[self.bsz, seq_len, self.kv_head_num, self.head_dim]);

        self.ctx_len = seq_len;

        for b in 0..self.bsz {
            for kh in 0..self.kv_head_num {
                let index = self.indexes[b][kh].clone();
                let keys = keys.slice(s![b, .., kh, ..]).to_owned();
                let vals = vals.slice(s![b, .., kh, ..]).to_owned();
                let neighbours = neighbours.slice(s![b, kh, .., ..]).to_owned();

                THREAD_POOL.execute(move || {
                    let mut index = index.write().unwrap();
                    index.prefill(keys.view(), vals.view(), neighbours.view()).unwrap();
                });
            }
        }

        Handle {
            result: InsertResult,
        }
    }

    // launch a thread for each batch and KV head to insert (async)
    // keys: (bsz, kv_head_num, head_dim)
    // vals: (bsz, kv_head_num, head_dim)
    // ep:   (bsz, kv_head_num, num_seeds)
    pub fn insert(&mut self, keys: ArrayView3<T>, vals: ArrayView3<T>, ep: ArrayView3<Neighbour>) -> Handle<InsertResult> {
        let num_seeds = ep.shape()[2];

        assert_eq!(keys.shape(), &[self.bsz, self.kv_head_num, self.head_dim]);
        assert_eq!(vals.shape(), &[self.bsz, self.kv_head_num, self.head_dim]);
        assert_eq!(ep.shape(),   &[self.bsz, self.kv_head_num, num_seeds]);

        let node_id = self.ctx_len as NodeID;
        self.ctx_len += 1;

        for b in 0..self.bsz {
            for kh in 0..self.kv_head_num {
                let index = self.indexes[b][kh].clone();
                let key = keys.slice(s![b, kh, ..]).to_owned();
                let val = vals.slice(s![b, kh, ..]).to_owned();
                let ep = ep.slice(s![b, kh, ..]).to_owned();

                THREAD_POOL.execute(move || {
                    let mut index = index.write().unwrap();
                    index.insert(node_id, key.view(), val.view(), ep.view()).unwrap();
                });
            }
        }

        Handle {
            result: InsertResult,
        }
    }

    // queries: (bsz, q_head_num, head_dim)
    // ep:      (bsz, kv_head_num, num_seeds)
    pub fn query(&self, queries: ArrayView3<T>, ef: usize, ep: ArrayView3<Neighbour>) -> Handle<QueryResult<T>> {
        let num_seeds = ep.shape()[2];

        assert_eq!(queries.shape(), &[self.bsz, self.q_head_num, self.head_dim]);
        assert_eq!(ep.shape(), &[self.bsz, self.kv_head_num, num_seeds]);

        let group_size = self.q_head_num / self.kv_head_num;

        let (tx, rx) = channel::unbounded();

        let search_bruteforce = self.opt.use_bruteforce;

        for b in 0..self.bsz {
            for kh in 0..self.kv_head_num {
                let index = self.indexes[b][kh].clone();
                let ep = ep.slice(s![b, kh, ..]).to_owned();
                let tx = tx.clone();
                let query = queries.slice(s![b, kh*group_size..(kh+1)*group_size, ..]).to_owned();
                THREAD_POOL.execute(move || {
                    let index = index.read().unwrap();
                    let (k_ret, v_ret) = if !search_bruteforce {
                        index.search(query.view(), ef, ep.view())
                    } else {
                        index.search_bruteforce(query.view(), ef)
                    };
                    tx.send((b, kh, k_ret, v_ret)).unwrap();
                });
            }
        }

        Handle {
            result: QueryResult {
                rx,
                bsz: self.bsz,
                head_num: self.kv_head_num,
                ef,
                head_dim: self.head_dim
            },
        }
    }

    pub fn save(&self, prefix: &str) {
        for b in 0..self.bsz {
            for kh in 0..self.kv_head_num {
                THREAD_POOL.execute({
                    let index = self.indexes[b][kh].clone();
                    let path = format!("{}b{:03}_h{:03}.idx", prefix, b, kh);
                    move || {
                        let index = index.read().unwrap();
                        index.save(&path).unwrap();
                    }
                });
            }
        }
    }

    pub fn load(&mut self, prefix: &str, ctx_len: usize) {
        self.ctx_len = ctx_len;
        for b in 0..self.bsz {
            for kh in 0..self.kv_head_num {
                THREAD_POOL.execute({
                    let index = self.indexes[b][kh].clone();
                    let path = format!("{}b{:03}_h{:03}.idx", prefix, b, kh);
                    move || {
                        let mut index = index.write().unwrap();
                        *index = nsw::Index::load(&path).unwrap();
                    }
                });
            }
        }
    }

    pub fn get_keys(&self) -> Array4<T> {
        let mut keys = Array4::<T>::from_elem((self.bsz, self.kv_head_num, self.ctx_len, self.head_dim), T::zero());
        
        for b in 0..self.bsz {
            for kh in 0..self.kv_head_num {
                let index = self.indexes[b][kh].clone();
                let index = index.read().unwrap();
                let key_view = index.get_keys(self.ctx_len);
                keys.slice_mut(s![b, kh, .., ..]).assign(&key_view);
            }
        }

        keys
    }
}

#[pyclass]
struct DiffCacheCPU {
    data_layer: DataLayer<bf16>,
}

#[pyclass]
struct InsertHandle {
    handle: Option<Handle<InsertResult>>,
}

#[pyclass]
struct QueryHandle {
    handle: Option<Handle<QueryResult<bf16>>>,
}

#[inline(always)]
fn view_u16_as_bf16<'a, D: Dimension>(v: ArrayView<'a, u16, D>) -> ArrayView<'a, bf16, D> {
    let ptr = v.as_ptr() as *const bf16;
    let shape = v.raw_dim();
    unsafe {
        ArrayView::from_shape_ptr(shape, ptr)
    }
}

fn owned_bf16_to_u16<D: Dimension>(a: Array<bf16, D>) -> Array<u16, D> {
    assert!(a.is_standard_layout(), "need standard C-order layout");
    let shape = a.raw_dim();
    let v_bf16: Vec<bf16> = a.into_raw_vec_and_offset().0;
    let (ptr, len, cap) = {
        let mut v = std::mem::ManuallyDrop::new(v_bf16);
        (v.as_mut_ptr() as *mut u16, v.len(), v.capacity())
    };
    Array::from_shape_vec(shape, unsafe { Vec::from_raw_parts(ptr, len, cap) }).unwrap()
}

fn setup_metrics() {
    let builder = TcpBuilder::new();
    builder.install().expect("failed to install TCP exporter");
}

#[pymethods]
impl DiffCacheCPU {
    #[new]
    #[pyo3(signature = (bsz, q_head_num, kv_head_num, head_dim, **kwargs))]
    fn new(bsz: usize, q_head_num: usize, kv_head_num: usize, head_dim: usize, kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        let mut opt = Options {
            max_ctx_len: 1000000,
            m: 16,
            ef_cons: 200,
            r_sq: [4.0].repeat(kv_head_num),
            batch_search: false,
            name: "l0".into(),
            use_bruteforce: true
        };
        if let Some(dict) = kwargs {
            for (key, value) in dict.iter() {
                match key.extract::<String>().unwrap() {
                    k if k == "max_ctx_len" => opt.max_ctx_len = value.extract::<usize>().unwrap(),
                    k if k == "m" => opt.m = value.extract::<usize>().unwrap(),
                    k if k == "ef_cons" => opt.ef_cons = value.extract::<usize>().unwrap(),
                    k if k == "r_sq" => opt.r_sq = value.extract::<Vec<f32>>().unwrap(),
                    k if k == "batch_search" => opt.batch_search = value.extract::<bool>().unwrap(),
                    k if k == "name" => opt.name = value.extract::<String>().unwrap(),
                    _ => panic!("Unknown option {key}"),
                }
            }
        }

        let data_layer = DataLayer::new(bsz, q_head_num, kv_head_num, head_dim, opt);

        DiffCacheCPU {
            data_layer
        }
    }

    fn insert<'py>(
        &mut self,
        keys: PyReadonlyArray3<'py, u16>,
        vals: PyReadonlyArray3<'py, u16>,
        ep_dists: PyReadonlyArray3<'py, f32>,
        ep_ids: PyReadonlyArray3<'py, u32>,
    ) -> InsertHandle {
        let keys = view_u16_as_bf16(keys.as_array());
        let vals = view_u16_as_bf16(vals.as_array());
        let ep_dists = ep_dists.as_array();
        let ep_ids = ep_ids.as_array();
        let ep = ndarray::Zip::from(&ep_dists)
            .and(&ep_ids)
            .map_collect(|&dist, &id| Neighbour { dist, node: id });

        let handle = self.data_layer.insert(keys, vals, ep.view());
        InsertHandle {
            handle: Some(handle)
        }
    }

    fn prefill<'py>(
        &mut self,
        keys: PyReadonlyArray4<'py, u16>,
        vals: PyReadonlyArray4<'py, u16>,
        neighbours: PyReadonlyArray4<'py, u32>,
    ) -> InsertHandle {
        let keys = view_u16_as_bf16(keys.as_array());
        let vals = view_u16_as_bf16(vals.as_array());
        let neighbours = neighbours.as_array();

        let handle = self.data_layer.prefill(keys, vals, neighbours);
        InsertHandle {
            handle: Some(handle)
        }
    }

    fn query<'py>(
        &self,
        queries: PyReadonlyArray3<'py, u16>,
        ef: usize,
        ep_dists: PyReadonlyArray3<'py, f32>,
        ep_ids: PyReadonlyArray3<'py, u32>,
    ) -> QueryHandle {
        let queries = view_u16_as_bf16(queries.as_array());
        let ep_dists = ep_dists.as_array();
        let ep_ids = ep_ids.as_array();
        let ep = ndarray::Zip::from(&ep_dists)
            .and(&ep_ids)
            .map_collect(|&dist, &id| Neighbour { dist, node: id });

        let handle = self.data_layer.query(queries, ef, ep.view());
        QueryHandle {
            handle: Some(handle)
        }
    }

    fn wait(&mut self) {
        THREAD_POOL.join();
    }

    fn save(&self, path: &str) {
        self.data_layer.save(path);
    }

    fn load(&mut self, path: &str, ctx_len: usize) {
        self.data_layer.load(path, ctx_len);
    }

    fn get_keys<'py>(&self, py: Python<'py>) -> Py<PyArray4<f32>> {
        let keys = self.data_layer.get_keys().mapv(|x| x.to_f32());
        keys.into_pyarray(py).into()
    }
}

#[pymethods]
impl InsertHandle {
    fn completed(&self) -> bool {
        self.handle.as_ref().unwrap().completed()
    }

    fn wait(&mut self) {
        let handle = self.handle.take().unwrap();
        let _ = handle.result();
    }
}

#[pymethods]
impl QueryHandle {
    fn completed(&self) -> bool {
        self.handle.as_ref().unwrap().completed()
    }

    fn collect<'py>(&mut self, py: Python<'py>) -> (Py<PyArray4<u16>>, Py<PyArray4<u16>>)
    {
        let handle = self.handle.take().expect("handle already taken");
        let (keys, vals) = handle.result().collect();
        let (keys, vals) = (owned_bf16_to_u16(keys), owned_bf16_to_u16(vals));

        (keys.into_pyarray(py).into(), vals.into_pyarray(py).into())
    }
}

#[pyfunction]
fn init_metrics() {
    setup_metrics();
}

#[pymodule]
fn _cpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DiffCacheCPU>()?;
    m.add_class::<InsertHandle>()?;
    m.add_class::<QueryHandle>()?;
    m.add_function(wrap_pyfunction!(init_metrics, m)?)?;
    Ok(())
}
