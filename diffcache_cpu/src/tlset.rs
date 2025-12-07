use thread_local::ThreadLocal;
use std::cell::{RefCell, RefMut};
use crate::HugeArray;

type IType = u16;

pub(crate) struct TLSet {
    size: usize,
    sets: ThreadLocal<RefCell<Set>>
}

pub(crate) struct Handle<'a> {
    set: RefMut<'a, Set>
}

struct Set {
    items: HugeArray<IType>,
    tag: IType
}

unsafe impl Send for Set {}
unsafe impl Sync for Set {}

impl Set {
    fn clear(&mut self) {
        if self.tag == IType::MAX {
            self.tag = 1;
            self.items.fill(0);
        } else {
            self.tag += 1;
        }
    }
}

impl TLSet {
    fn get_set(&'_ self) -> RefMut<'_, Set> {
        self.sets.get_or(|| {
            RefCell::new(Set {
                items: HugeArray::new_with(self.size, 0),
                tag: 0
            })
        }).borrow_mut()
    }

    pub(crate) fn get_handle(&'_ self) -> Handle<'_> {
        let mut set = self.get_set();
        set.clear();
        Handle { set }
    }

    pub(crate) fn new(size: usize) -> Self {
        TLSet {
            size,
            sets: ThreadLocal::new()
        }
    }
}

impl<'a> Handle<'a> {
    #[inline]
    pub(crate) fn insert(&mut self, item: usize) {
        self.set.items[item] = self.set.tag;
    }

    #[inline]
    pub(crate) fn prefetch(&self, item: usize) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let ptr = self.set.items.as_ptr().add(item) as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let _ = (&self, item);
        }
    }

    #[inline]
    pub(crate) fn contains(&self, item: usize) -> bool {
        self.set.items[item] == self.set.tag
    }
}
