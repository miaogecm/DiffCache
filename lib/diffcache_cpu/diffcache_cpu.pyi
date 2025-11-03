# diffcache_cpu.pyi
from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

__all__ = ["DiffCacheCPU", "InsertHandle", "QueryHandle"]

class DiffCacheCPU:
    """
    CPU-side data layer for DiffCache.

    Shape conventions (C-order arrays):
      - keys / vals : (bsz, kv_head_num, head_dim)
      - queries     : (bsz, q_head_num,  head_dim)
      - ep_dists    : (bsz, kv_head_num or q_head_num, num_seeds)
      - ep_ids      : same as ep_dists
    """

    def __init__(
        self,
        bsz: int,
        q_head_num: int,
        kv_head_num: int,
        head_dim: int,
        *,
        max_ctx_len: int = 1_000_000,
        m: int = 16,
        ef_cons: int = 200,
        thread_pool_size: int = 16,
        r_sq: float = 4.0,
        group_query: bool = True,
    ) -> None: ...
    """
    Create a DiffCacheCPU instance.
    """

    def insert(
        self,
        keys: NDArray[np.uint16],
        vals: NDArray[np.uint16],
        ep_dists: NDArray[np.float32],
        ep_ids: NDArray[np.uint32],
    ) -> InsertHandle: ...
    """
    Insert one step of K/V into the cache.

    Parameters
    ----------
    keys, vals : uint16 arrays, shape (bsz, kv_head_num, head_dim)
        BF16 bit-patterns stored as uint16.
    ep_dists   : float32 array, shape (bsz, kv_head_num, num_seeds)
    ep_ids     : uint32 array,  shape (bsz, kv_head_num, num_seeds)

    Returns
    -------
    InsertHandle
    """

    def query(
        self,
        queries: NDArray[np.uint16],
        ef: int,
        ep_dists: NDArray[np.float32],
        ep_ids: NDArray[np.uint32],
    ) -> QueryHandle: ...
    """
    Retrieve top-ef relevant K/V for given queries.

    Parameters
    ----------
    queries  : uint16 array, shape (bsz, q_head_num, head_dim)
        BF16 bit-patterns stored as uint16.
    ep_*     : when group_query=True, shape (bsz, kv_head_num, num_seeds);
               otherwise shape (bsz, q_head_num, num_seeds)

    Returns
    -------
    QueryHandle
    """


class InsertHandle:
    """Asynchronous handle returned by DiffCacheCPU.insert()."""

    def completed(self) -> bool: ...
    """Return True if all insert tasks have finished."""

    def wait(self) -> None: ...
    """Block until insertion is complete."""


class QueryHandle:
    """Asynchronous handle returned by DiffCacheCPU.query()."""

    def completed(self) -> bool: ...
    """Return True if all query tasks have finished."""

    def collect(self) -> Tuple[NDArray[np.uint16], NDArray[np.uint16]]: ...
    """
    Gather results: returns (keys, vals), both uint16 arrays.

    Shapes
    ------
    (bsz, ef, head_num, head_dim), where:
      - head_num = kv_head_num  if group_query=True
      - head_num = q_head_num   if group_query=False
    """
