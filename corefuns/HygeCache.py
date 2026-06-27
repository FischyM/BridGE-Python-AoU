import functools

import numpy as np
from scipy.stats import hypergeom

def _hyge_chunk(arg):
    n, d, g_chunk, x_chunk = arg
    """Vectorized worker — no Python-level loop, no lru_cache needed."""
    g_chunk = np.rint(g_chunk).astype(np.int64)
    x_chunk = np.rint(x_chunk).astype(np.int64)
    return hypergeom.sf(x_chunk - 1, n, g_chunk, d)

class HygeCache:
    def __init__(self, sample_size, case_size, control_size):
        self.sample_size = sample_size
        self.case_size = case_size
        self.control_size = control_size

    def apply_hyge(self, g, x, case_flag, pool, n_workers):
        d = self.case_size if case_flag else self.control_size
        g_chunks = np.array_split(g, n_workers)
        x_chunks = np.array_split(x, n_workers)
        args = [(self.sample_size, d, gc, xc) for gc, xc in zip(g_chunks, x_chunks)]
        results = pool.map(_hyge_chunk, args)
        return np.concatenate(results)
