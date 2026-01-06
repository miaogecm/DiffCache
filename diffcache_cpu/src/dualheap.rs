use std::collections::BinaryHeap;

// maintains the smallest `cap` elements, supports querying the d-th smallest (1-indexed)
pub(crate) struct DualHeap<T> {
    pub(crate) small: BinaryHeap<T>, // max-heap: holds the smallest d elements
    pub(crate) rest:  BinaryHeap<T>, // max-heap: holds the next (cap - d) elements
    pub(crate) small_replace_count: usize,
    d: usize,
    cap: usize,
}

impl<T: Ord> DualHeap<T> {
    pub(crate) fn new(d: usize, cap: usize) -> Self {
        assert!(d >= 1, "d must be >= 1");
        assert!(cap >= d, "cap must be >= d");
        Self {
            small: BinaryHeap::with_capacity(d),
            rest:  BinaryHeap::with_capacity(cap - d),
            small_replace_count: 0,
            d,
            cap,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.small.len() + self.rest.len()
    }

    // the d-th smallest (1-indexed). Returns None if len < d.
    pub(crate) fn dth_min(&self) -> Option<&T> {
        if self.len() < self.d { None } else { self.small.peek() }
    }

    // the maximum of the kept elements. Returns None if len == 0.
    pub(crate) fn kept_max(&self) -> Option<&T> {
        self.rest.peek().or_else(|| self.small.peek())
    }

    // push into DualHeap
    pub(crate) fn push(&mut self, item: T) {
        // still have space in small, push into small
        if self.small.len() < self.d {
            self.small.push(item);
            return;
        }

        // there is no space in small, push into rest
        // if within small range, push into small and move the largest in small from small to rest
        let mut item = item;
        if self.small.peek().unwrap() > &item {
            self.small_replace_count += 1;
            item = std::mem::replace(&mut *self.small.peek_mut().unwrap(), item);
        }

        // no rest case
        if self.cap == self.d {
            return;
        }

        // rest has space, push into rest directly
        if self.rest.len() < self.cap - self.d {
            self.rest.push(item);
            return;
        }

        // rest is full, item is too large, discard it
        if self.rest.peek().unwrap() <= &item {
            return;
        }

        // item can be inserted into rest, push into rest and pop the largest
        *self.rest.peek_mut().unwrap() = item;
    }

    pub(crate) fn from_iter_with<I: IntoIterator<Item = T>>(d: usize, cap: usize, iter: I) -> Self {
        let mut h = DualHeap::new(d, cap);
        h.extend(iter);
        h
    }

    pub(crate) fn into_sorted_vec(self) -> Vec<T> {
        let mut v = self.small.into_sorted_vec();
        let mut r = self.rest.into_sorted_vec();
        v.append(&mut r);
        v
    }
}

impl<T: Ord> Extend<T> for DualHeap<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T: Ord + Clone> Clone for DualHeap<T> {
    fn clone(&self) -> Self {
        Self {
            small: self.small.clone(),
            rest:  self.rest.clone(),
            small_replace_count: self.small_replace_count,
            d: self.d,
            cap: self.cap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    // ---- helpers ----

    fn baseline_kept(all: &[i32], cap: usize) -> Vec<i32> {
        let mut v = all.to_vec();
        v.sort();                 // ascending
        v.truncate(cap);          // smallest cap
        v
    }

    fn baseline_dth(all: &[i32], d: usize, cap: usize) -> Option<i32> {
        let kept = baseline_kept(all, cap);
        if kept.len() < d { None } else { Some(kept[d - 1]) } // 1-indexed d-th smallest
    }

    fn baseline_kept_max(all: &[i32], cap: usize) -> Option<i32> {
        let kept = baseline_kept(all, cap);
        kept.last().copied()
    }

    fn heap_to_sorted_vec(h: &BinaryHeap<i32>) -> Vec<i32> {
        h.clone().into_sorted_vec() // ascending
    }

    fn dualheap_kept_sorted(h: &DualHeap<i32>) -> Vec<i32> {
        let mut a = heap_to_sorted_vec(&h.small);
        let mut b = heap_to_sorted_vec(&h.rest);
        a.append(&mut b);
        debug_assert!(a.windows(2).all(|w| w[0] <= w[1]));
        a
    }

    fn assert_invariants(h: &DualHeap<i32>) {
        assert!(h.small.len() <= h.d, "small.len must be <= d");
        assert!(h.rest.len() <= h.cap - h.d, "rest.len must be <= cap-d");
        if h.cap == h.d {
            assert!(h.rest.is_empty(), "cap==d => rest should be empty");
        }
    }

    fn assert_matches_baseline(h: &DualHeap<i32>, all: &[i32]) {
        assert_invariants(h);

        // kept elements (sorted)
        let kept = dualheap_kept_sorted(h);
        let base_kept = baseline_kept(all, h.cap);

        assert_eq!(kept, base_kept, "kept set mismatch");

        // dth_min
        let got_dth = h.dth_min().copied();
        let exp_dth = baseline_dth(all, h.d, h.cap);
        assert_eq!(got_dth, exp_dth, "dth_min mismatch");

        // kept_max
        let got_kmax = h.kept_max().copied();
        let exp_kmax = baseline_kept_max(all, h.cap);
        assert_eq!(got_kmax, exp_kmax, "kept_max mismatch");
    }

    // simple deterministic pseudo RNG (LCG), no external crate needed
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self { Self(seed) }
        fn next_u32(&mut self) -> u32 {
            // constants from Numerical Recipes-ish
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 32) as u32
        }
        fn next_i32_range(&mut self, lo: i32, hi: i32) -> i32 {
            let r = self.next_u32();
            let span = (hi - lo + 1) as u32;
            lo + (r % span) as i32
        }
    }

    // ---- tests ----

    #[test]
    fn test_len_less_than_d() {
        let d = 5;
        let cap = 10;
        let mut h = DualHeap::new(d, cap);

        let mut all = Vec::<i32>::new();
        for &x in &[7, 3, 9, 1] { // only 4 < d
            h.push(x);
            all.push(x);

            assert_eq!(h.dth_min().copied(), None);
            assert_matches_baseline(&h, &all);
        }
    }

    #[test]
    fn test_d_equals_cap_no_rest() {
        let d = 5;
        let cap = 5;
        let mut h = DualHeap::new(d, cap);

        let mut all = Vec::<i32>::new();

        // fill
        for &x in &[10, 9, 8, 7, 6] {
            h.push(x);
            all.push(x);
            assert_matches_baseline(&h, &all);
        }

        // push a larger item: should be discarded (kept are smallest 5)
        h.push(100);
        all.push(100);
        assert_matches_baseline(&h, &all);

        // push a smaller item: should replace the current kept_max
        h.push(1);
        all.push(1);
        assert_matches_baseline(&h, &all);

        // into_sorted_vec should match baseline kept
        let got = h.into_sorted_vec();
        let exp = baseline_kept(&all, cap);
        assert_eq!(got, exp);
    }

    #[test]
    fn test_cap_greater_than_d_basic_behaviors() {
        let d = 3;
        let cap = 6;
        let mut h = DualHeap::new(d, cap);
        let mut all = Vec::<i32>::new();

        // Insert a mix that exercises: small fill, small replace, rest fill, rest discard/replace
        let seq = [50, 40, 30, 20, 60, 10, 70, 25, 5, 100, 15];
        for &x in &seq {
            h.push(x);
            all.push(x);
            assert_matches_baseline(&h, &all);
        }

        // spot-check dth_min logic
        let got_dth = h.dth_min().copied();
        let exp_dth = baseline_dth(&all, d, cap);
        assert_eq!(got_dth, exp_dth);
    }

    #[test]
    fn test_duplicates_and_ties() {
        let d = 4;
        let cap = 7;
        let mut h = DualHeap::new(d, cap);
        let mut all = Vec::<i32>::new();

        // lots of duplicates
        let seq = [5, 5, 5, 5, 4, 4, 6, 6, 3, 3, 3, 7, 2, 2];
        for &x in &seq {
            h.push(x);
            all.push(x);
            assert_matches_baseline(&h, &all);
        }

        // into_sorted_vec should be ascending and match baseline
        let got = h.into_sorted_vec();
        let exp = baseline_kept(&all, cap);
        assert_eq!(got, exp);
        assert!(got.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_extend_and_from_iter_with() {
        let d = 5;
        let cap = 10;
        let seq: Vec<i32> = vec![9, 1, 8, 2, 7, 3, 6, 4, 5, 0, 10, -1];

        // from_iter_with
        let h1 = DualHeap::from_iter_with(d, cap, seq.iter().copied());
        let mut all = seq.clone();
        assert_matches_baseline(&h1, &all);

        // extend
        let mut h2 = DualHeap::new(d, cap);
        h2.extend(seq.iter().copied());
        assert_matches_baseline(&h2, &all);

        // both should yield same sorted vec
        let v1 = h1.into_sorted_vec();
        let v2 = h2.into_sorted_vec();
        let exp = baseline_kept(&all, cap);
        assert_eq!(v1, exp);
        assert_eq!(v2, exp);
    }

    #[test]
    fn test_random_regression_multiple_configs() {
        let configs = [
            (1, 1),
            (1, 10),
            (3, 3),
            (3, 10),
            (10, 10),
            (10, 100),
            (100, 100),
            (100, 1000),
        ];

        let mut rng = Lcg::new(0xC0FFEE_u64);

        for &(d, cap) in &configs {
            let mut h = DualHeap::new(d, cap);
            let mut all = Vec::<i32>::new();

            // keep it moderate; we verify every step
            for _ in 0..300 {
                let x = rng.next_i32_range(-1000, 1000);
                h.push(x);
                all.push(x);

                assert_matches_baseline(&h, &all);
            }

            // final into_sorted_vec check
            let got = h.into_sorted_vec();
            let exp = baseline_kept(&all, cap);
            assert_eq!(got, exp);
        }
    }

    #[test]
    fn test_kept_max_semantics_when_not_full() {
        // kept_max should be the max of currently kept elements even before reaching cap
        let d = 4;
        let cap = 10;
        let mut h = DualHeap::new(d, cap);
        let mut all = Vec::<i32>::new();

        for &x in &[3, 1, 4, 1, 5] {
            h.push(x);
            all.push(x);

            let got = h.kept_max().copied();
            let exp = all.iter().max().copied(); // since not full, kept == all
            assert_eq!(got, exp);

            assert_matches_baseline(&h, &all);
        }
    }
}
