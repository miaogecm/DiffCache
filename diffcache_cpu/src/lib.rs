use std::{cmp::min, ops::{Add, Deref, DerefMut, Div}, sync::{Arc, atomic::{AtomicBool, AtomicUsize}}, time::Instant};
use metrics::counter;
use ndarray::{Array, Array1, Array2, Array3, Array4, ArrayView, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut, ArrayViewMut4, Axis, Data, Dimension, Ix4, ShapeBuilder, Zip, s};
use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadwriteArray4};
use simsimd::{bf16, SpatialSimilarity};
use std::sync::RwLock;
use pyo3::{prelude::*, types::PyDict};
use metrics_exporter_tcp::TcpBuilder;
use rayon::prelude::*;

mod dualheap;
mod tlset;
mod nsw;

pub type Distance = f32;

type NodeID = u32;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[inline(always)]
pub(crate) fn prefetch_ptr<T, const LOCALITY: i32>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{
            _mm_prefetch, _MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2,
        };
        if LOCALITY <= 0 {
            _mm_prefetch(ptr as *const i8, _MM_HINT_NTA);
        } else if LOCALITY == 1 {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T2);
        } else if LOCALITY == 2 {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
        } else {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        };
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

pub trait InnerProduct {
    fn dot(a: &[Self], b: &[Self]) -> f32 where Self: Sized;
}

pub trait L2Square {
    fn l2sq(a: &[Self], b: &[Self]) -> f32 where Self: Sized;
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
    use_bruteforce: bool,
    attn_mass_threshold: f32,
    k_check_seq: Vec<usize>,
}

pub struct DataLayer<T: Clone> {
    indexes: Vec<nsw::Index<T>>,                         // per batch per KV head index
    ctx_len: usize,                                      // current context length
    bsz: usize,                                          // batch size
    kv_head_num: usize,                                  // number of KV heads
    q_head_num: usize,                                   // number of Q heads
    head_dim: usize,                                     // dimension per head
    opt: Options,
}

impl<T: 'static + Send + Sync + Clone + Value + Copy + InnerProduct + L2Square> DataLayer<T> {
    pub fn new(bsz: usize, q_head_num: usize, kv_head_num: usize, head_dim: usize, opt: Options) -> Self {
        assert!(q_head_num % kv_head_num == 0, "q_head_num must be multiple of kv_head_num");

        let mut indexes = vec![];

        for _ in 0..bsz {
            for h in 0..kv_head_num {
                indexes.push(nsw::Index::new(
                    head_dim,
                    nsw::Options {
                        n_max: opt.max_ctx_len,
                        m: opt.m,
                        ef_cons: opt.ef_cons,
                        r_sq: opt.r_sq[h],
                        name: format!("{}b{:03}h{:03}", opt.name, indexes.len(), h),
                        attn_mass_threshold: opt.attn_mass_threshold,
                        k_check_seq: opt.k_check_seq.clone(),
                    }
                ));
            }
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

    // keys: (bsz, kv_head_num, seq_len, head_dim)
    // vals: (bsz, kv_head_num, seq_len, head_dim)
    // neighbours: (bsz, num_kv_heads, seq_len, deg) u32
    pub fn prefill(&mut self, keys: ArrayView4<T>, vals: ArrayView4<T>, neighbours: ArrayView4<u32>) {
        let seq_len = keys.shape()[2];

        assert_eq!(keys.shape(), &[self.bsz, self.kv_head_num, seq_len, self.head_dim]);
        assert_eq!(vals.shape(), &[self.bsz, self.kv_head_num, seq_len, self.head_dim]);

        self.ctx_len = seq_len;

        self.indexes.par_iter_mut().enumerate().for_each(|(idx, index)| {
            let b = idx / self.kv_head_num;
            let kh = idx % self.kv_head_num;

            let keys = keys.slice(s![b, kh, .., ..]);
            let vals = vals.slice(s![b, kh, .., ..]);
            let neighbours = neighbours.slice(s![b, kh, .., ..]);

            index.prefill(keys, vals, neighbours).unwrap();
        });
    }

    // launch a thread for each batch and KV head to insert (async)
    // keys: (bsz, kv_head_num, head_dim)
    // vals: (bsz, kv_head_num, head_dim)
    pub fn insert(&mut self, keys: ArrayView3<T>, vals: ArrayView3<T>) {
        assert_eq!(keys.shape(), &[self.bsz, self.kv_head_num, self.head_dim]);
        assert_eq!(vals.shape(), &[self.bsz, self.kv_head_num, self.head_dim]);

        self.ctx_len += 1;

        let ep = ArrayView1::<Neighbour>::from(&[]);

        self.indexes.par_iter_mut().enumerate().for_each(|(idx, index)| {
            let b = idx / self.kv_head_num;
            let kh = idx % self.kv_head_num;

            let key = keys.slice(s![b, kh, ..]);
            let val = vals.slice(s![b, kh, ..]);

            index.insert(key, val, ep).unwrap();
        });
    }

    // out:     (bsz, kv_head_num, ef, head_dim)
    // queries: (bsz, kv_head_num, head_dim)
    pub fn query(&self, out_keys: ArrayViewMut4<T>, out_vals: ArrayViewMut4<T>, queries: ArrayView3<T>, ef: usize) {
        let ef = min(ef, self.ctx_len);

        assert_eq!(queries.shape(), &[self.bsz, self.kv_head_num, self.head_dim]);

        let search_bruteforce = self.opt.use_bruteforce;

        let ep = ArrayView1::<Neighbour>::from(&[]);

        let start = Instant::now();

        let mut flat_keys = out_keys.into_shape_with_order((self.bsz * self.kv_head_num, ef, self.head_dim)).unwrap();
        let mut flat_vals = out_vals.into_shape_with_order((self.bsz * self.kv_head_num, ef, self.head_dim)).unwrap();

        flat_keys.axis_iter_mut(Axis(0)).into_par_iter().zip(flat_vals.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .for_each(|(idx, (out_keys_slice, out_vals_slice))| {
                let b = idx / self.kv_head_num;
                let kh = idx % self.kv_head_num;

                let query = queries.slice(s![b, kh, ..]);

                let index = &self.indexes[b * self.kv_head_num + kh];
                if !search_bruteforce {
                    index.search(query, ef, ep, out_keys_slice, out_vals_slice);
                } else {
                    index.search_bruteforce(query, ef, out_keys_slice, out_vals_slice);
                }
            });

        counter!(format!("{}_total_query_latency", self.opt.name)).increment(start.elapsed().as_micros() as u64);
    }

    pub fn save(&self, prefix: &str) {
        let tasks: Vec<(usize, usize)> = (0..self.bsz)
            .flat_map(|b| (0..self.kv_head_num).map(move |kh| (b, kh)))
            .collect();

        tasks.into_par_iter().for_each(|(b, kh)| {
            let index = &self.indexes[b * self.kv_head_num + kh];
            let path = format!("{}b{:03}_h{:03}.idx", prefix, b, kh);
            index.save(&path).unwrap();
        });
    }

    pub fn load(&mut self, prefix: &str, ctx_len: usize) {
        self.ctx_len = ctx_len;

        self.indexes.par_iter_mut().enumerate().for_each(|(idx, index)| {
            let b = idx / self.kv_head_num;
            let kh = idx % self.kv_head_num;

            let path = format!("{}b{:03}_h{:03}.idx", prefix, b, kh);
            *index = nsw::Index::load(&path).unwrap();
        });
    }

    pub fn get_keys(&self) -> Array4<T> {
        let mut keys = Array4::<T>::from_elem((self.bsz, self.kv_head_num, self.ctx_len, self.head_dim), T::zero());
        
        for b in 0..self.bsz {
            for kh in 0..self.kv_head_num {
                let index = &self.indexes[b * self.kv_head_num + kh];
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

#[inline(always)]
fn view_u16_as_bf16<'a, D: Dimension>(v: ArrayView<'a, u16, D>) -> ArrayView<'a, bf16, D> {
    let ptr = v.as_ptr() as *const bf16;
    let shape = v.raw_dim();
    unsafe {
        ArrayView::from_shape_ptr(shape, ptr)
    }
}

#[inline(always)]
fn view_u16_as_bf16_mut<'a, D: Dimension>(mut v: ArrayViewMut<'a, u16, D>) -> ArrayViewMut<'a, bf16, D> {
    let ptr = v.as_mut_ptr() as *mut bf16;
    let shape = v.raw_dim();
    unsafe {
        ArrayViewMut::from_shape_ptr(shape, ptr)
    }
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
            use_bruteforce: false,
            attn_mass_threshold: 0.9,
            k_check_seq: vec![100, 200, 300, 400, 800, 1600, 3200, 6400, 12800, 25600],
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
                    k if k == "use_bruteforce" => opt.use_bruteforce = value.extract::<bool>().unwrap(),
                    k if k == "attn_mass_threshold" => opt.attn_mass_threshold = value.extract::<f32>().unwrap(),
                    k if k == "k_check_seq" => opt.k_check_seq = value.extract::<Vec<usize>>().unwrap(),
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
        vals: PyReadonlyArray3<'py, u16>
    ) {
        let keys = view_u16_as_bf16(keys.as_array());
        let vals = view_u16_as_bf16(vals.as_array());
        self.data_layer.insert(keys, vals);
    }

    fn prefill<'py>(
        &mut self,
        keys: PyReadonlyArray4<'py, u16>,
        vals: PyReadonlyArray4<'py, u16>,
        neighbours: PyReadonlyArray4<'py, u32>,
    ) {
        let keys = view_u16_as_bf16(keys.as_array());
        let vals = view_u16_as_bf16(vals.as_array());
        let neighbours = neighbours.as_array();
        self.data_layer.prefill(keys, vals, neighbours);
    }

    fn query<'py>(
        &self,
        mut out_keys: PyReadwriteArray4<'py, u16>,
        mut out_vals: PyReadwriteArray4<'py, u16>,
        queries: PyReadonlyArray3<'py, u16>,
        ef: usize,
    ) {
        let out_keys = view_u16_as_bf16_mut(out_keys.as_array_mut());
        let out_vals = view_u16_as_bf16_mut(out_vals.as_array_mut());
        let queries = view_u16_as_bf16(queries.as_array());
        self.data_layer.query(out_keys, out_vals, queries, ef);
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

    fn update_r_sq(&mut self, r_sq: Vec<f32>) {
        assert!(r_sq.len() == self.data_layer.kv_head_num, "r_sq length must match kv_head_num");

        self.data_layer.indexes.par_iter_mut().enumerate().for_each(|(idx, index)| {
            let kh = idx % self.data_layer.kv_head_num;
            let r_sq = r_sq[kh];
            index.update_r_sq(r_sq);
        });
    }
}

#[pyfunction]
fn init_metrics() {
    setup_metrics();
}

#[pymodule]
fn _cpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DiffCacheCPU>()?;
    m.add_function(wrap_pyfunction!(init_metrics, m)?)?;
    Ok(())
}
