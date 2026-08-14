import functools

import numpy as np
from scipy.stats import hypergeom


# HYGETEST computes Hypergeometric cumulative distribution for
# the given inputs.
#
# INPUTS:
# n - is the population size
# d - is the number of draws
# k - is the number of successes
# m - is the number of success states in the population
#
# OUTPUTS:
# logpv - negative log10(p-value)
# pv - p-value

@functools.lru_cache(maxsize=None)  # TODO: let it grow to max size? theoretically there will be a ceiling...?
def _hyge_single(n, d, g, x):
    """Single-value hypergeom survival function, cached per (n, d, g, x)."""
    return hypergeom.sf(x - 1, n, g, d)

def _hyge_chunk(arg):
    n, d, g_chunk, x_chunk = arg
    out = np.empty(len(g_chunk), dtype=np.float64)
    for i in range(len(g_chunk)):
        out[i] = _hyge_single(n, d, g_chunk[i], x_chunk[i])
    return out

class HygeCache:
    """Drop-in replacement for HygeCache, but dispatches one value at a time
    per worker with functools.lru_cache, instead of vectorized hypergeom.sf calls.

    Caching only pays off if (n, d, g, x) tuples repeat often within a worker's
    lifetime across calls sharing the same pool. Since g/x are SNP pair counts
    bounded by population/case size, repeats are plausible after int rounding -
    but if most tuples are unique, the per-value Python loop + cache-miss overhead
    will likely be slower than the vectorized scipy call. Worth benchmarking both
    on real data before swapping in.
    """

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

    @staticmethod
    def cache_info():
        """Inspect hit/miss rate to decide if caching is actually helping (call inside
        a worker, e.g. via pool.apply, since each process has its own cache)."""
        return _hyge_single.cache_info()
