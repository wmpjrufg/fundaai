"""Tests for ``core.optimization.cache`` (surrogate-model cache).

The cache is a content-addressed LRU that stores fitted Gaussian
Process Regressor pipelines so repeated calls with the same training
data and configuration short-circuit the expensive kernel
hyperparameter optimisation. These tests lock the contract on three
fronts:

    1. **Fingerprint stability**: the same (X, y, pipeline) yields the
       same key bit-for-bit; changing any field changes the key.
    2. **Cache semantics**: hits return predictions equal to a fresh
       fit; misses miss; LRU eviction is in place; clear() resets
       state.
    3. **Optimiser integration**: ``fit_or_get_cached`` is a drop-in
       replacement for ``pipe.fit`` and ``ego_01_architecture`` runs
       end-to-end with the cache enabled, returning the same objective
       value as without the cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.optimization import benchmark
from core.optimization.cache import (
    SurrogateCache,
    fingerprint,
    fit_or_get_cached,
    pipeline_signature,
)
from core.optimization.ego import ego_01_architecture


# =============================================================================
# Helpers
# =============================================================================
def _make_pipeline(kernel=None, random_state=42, alpha=0.1):
    """This helper builds a fresh GPR pipeline mirroring the EGO setup.

    :param kernel: Optional kernel; defaults to RBF()
    :param random_state: GPR random_state
    :param alpha: GPR alpha (jitter)

    :return: Unfitted scikit-learn Pipeline
    """
    if kernel is None:
        kernel = RBF()
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gp",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    alpha=alpha,
                    n_restarts_optimizer=2,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _make_data(n=10, d=2, seed=0):
    """This helper builds a synthetic (X, y) DataFrame pair.

    :param n: Number of rows
    :param d: Number of features
    :param seed: RNG seed for reproducibility

    :return: Tuple (X DataFrame, y DataFrame)
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.uniform(-1, 1, size=(n, d)), columns=[f"X_{i}" for i in range(d)])
    y = pd.DataFrame((X.to_numpy() ** 2).sum(axis=1), columns=["OF"])
    return X, y


# =============================================================================
# pipeline_signature
# =============================================================================
@pytest.mark.optimization
class TestPipelineSignature:
    """This class verifies that pipeline_signature is stable and discriminating."""

    def test_same_pipeline_same_signature(self):
        """This test ensures two identical pipelines produce the same signature."""
        a = _make_pipeline()
        b = _make_pipeline()
        assert pipeline_signature(a) == pipeline_signature(b)

    def test_different_kernel_different_signature(self):
        """This test ensures swapping the kernel changes the signature."""
        a = _make_pipeline(kernel=RBF())
        b = _make_pipeline(kernel=Matern(nu=1.5))
        assert pipeline_signature(a) != pipeline_signature(b)

    def test_different_random_state_different_signature(self):
        """This test ensures random_state is part of the signature."""
        a = _make_pipeline(random_state=42)
        b = _make_pipeline(random_state=7)
        assert pipeline_signature(a) != pipeline_signature(b)

    def test_different_alpha_different_signature(self):
        """This test ensures alpha is part of the signature."""
        a = _make_pipeline(alpha=0.1)
        b = _make_pipeline(alpha=0.01)
        assert pipeline_signature(a) != pipeline_signature(b)


# =============================================================================
# fingerprint
# =============================================================================
@pytest.mark.optimization
class TestFingerprint:
    """This class verifies the SHA-256 fingerprint contract."""

    def test_same_inputs_same_key(self):
        """This test ensures identical (X, y, signature) produce the same digest."""
        X, y = _make_data()
        sig = pipeline_signature(_make_pipeline())
        assert fingerprint(X, y, sig) == fingerprint(X, y, sig)

    def test_perturb_y_changes_key(self):
        """This test ensures any change in y produces a different digest."""
        X, y = _make_data()
        sig = pipeline_signature(_make_pipeline())
        y2 = y.copy()
        y2.iloc[0, 0] += 1e-9
        assert fingerprint(X, y, sig) != fingerprint(X, y2, sig)

    def test_perturb_X_changes_key(self):
        """This test ensures any change in X produces a different digest."""
        X, y = _make_data()
        sig = pipeline_signature(_make_pipeline())
        X2 = X.copy()
        X2.iloc[0, 0] += 1e-9
        assert fingerprint(X, y, sig) != fingerprint(X2, y, sig)

    def test_signature_change_changes_key(self):
        """This test ensures the signature factor is honoured."""
        X, y = _make_data()
        sig_a = pipeline_signature(_make_pipeline(kernel=RBF()))
        sig_b = pipeline_signature(_make_pipeline(kernel=Matern(nu=1.5)))
        assert fingerprint(X, y, sig_a) != fingerprint(X, y, sig_b)

    def test_dataframe_and_array_inputs_match(self):
        """This test ensures pd.DataFrame and ndarray inputs produce the same digest.

        The cache is fed by EGO with DataFrames, but the cache module
        coerces both to float64 contiguous ndarrays. The digest must
        therefore be invariant to the wrapper type as long as the
        values match.
        """
        X, y = _make_data()
        sig = pipeline_signature(_make_pipeline())
        d1 = fingerprint(X, y, sig)
        d2 = fingerprint(X.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64), sig)
        assert d1 == d2


# =============================================================================
# SurrogateCache (in-memory)
# =============================================================================
@pytest.mark.optimization
class TestSurrogateCacheMemory:
    """This class verifies the LRU semantics of SurrogateCache without disk."""

    def test_miss_returns_none(self):
        """This test ensures get on an empty cache returns None and bumps misses."""
        cache = SurrogateCache(maxsize=4)
        assert cache.get("nope") is None
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0

    def test_put_then_get_hits(self):
        """This test ensures put followed by get returns a deep copy and counts a hit."""
        cache = SurrogateCache(maxsize=4)
        X, y = _make_data()
        pipe = _make_pipeline()
        pipe.fit(X, y)
        cache.put("k", pipe)
        out = cache.get("k")
        assert out is not None
        assert cache.stats["hits"] == 1
        # Predictions match
        np.testing.assert_allclose(out.predict(X), pipe.predict(X), rtol=0, atol=0)
        # Different object (deep-copy semantics on read)
        assert out is not pipe

    def test_lru_eviction_evicts_oldest(self):
        """This test ensures exceeding maxsize evicts the least-recently-used key."""
        cache = SurrogateCache(maxsize=2)
        X, y = _make_data()
        pipe = _make_pipeline()
        pipe.fit(X, y)
        cache.put("a", pipe)
        cache.put("b", pipe)
        cache.put("c", pipe)
        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache

    def test_get_promotes_to_mru(self):
        """This test ensures get on an existing key prevents its eviction."""
        cache = SurrogateCache(maxsize=2)
        X, y = _make_data()
        pipe = _make_pipeline()
        pipe.fit(X, y)
        cache.put("a", pipe)
        cache.put("b", pipe)
        cache.get("a")          # promote a -> MRU
        cache.put("c", pipe)    # b should be evicted, not a
        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache

    def test_clear_resets_state_and_stats(self):
        """This test ensures clear empties the cache and resets counters."""
        cache = SurrogateCache(maxsize=2)
        X, y = _make_data()
        pipe = _make_pipeline()
        pipe.fit(X, y)
        cache.put("a", pipe)
        cache.get("a")
        cache.clear()
        assert len(cache) == 0
        assert cache.stats == {"hits": 0, "misses": 0, "disk_hits": 0, "size": 0}

    def test_maxsize_must_be_positive(self):
        """This test ensures maxsize=0 (or negative) raises."""
        with pytest.raises(ValueError):
            SurrogateCache(maxsize=0)


# =============================================================================
# SurrogateCache (disk-backed)
# =============================================================================
@pytest.mark.optimization
class TestSurrogateCacheDisk:
    """This class verifies disk persistence and rehydration."""

    def test_disk_rehydrates_after_clear(self, tmp_path):
        """This test ensures clear() does not wipe disk; the next get() rehydrates."""
        cache = SurrogateCache(maxsize=2, disk_dir=tmp_path)
        X, y = _make_data()
        pipe = _make_pipeline()
        pipe.fit(X, y)
        cache.put("k", pipe)
        cache.clear()
        # Memory miss + disk hit
        out = cache.get("k")
        assert out is not None
        assert cache.stats["disk_hits"] == 1
        np.testing.assert_allclose(out.predict(X), pipe.predict(X), rtol=0, atol=0)

    def test_disk_dir_is_created(self, tmp_path):
        """This test ensures the disk directory is created on demand."""
        nested = tmp_path / "deep" / "cache"
        SurrogateCache(maxsize=2, disk_dir=nested)
        assert nested.is_dir()


# =============================================================================
# fit_or_get_cached
# =============================================================================
@pytest.mark.optimization
class TestFitOrGetCached:
    """This class verifies the high-level fit/lookup helper."""

    def test_no_cache_falls_back_to_fit(self):
        """This test ensures cache=None reproduces plain pipe.fit(X, y)."""
        X, y = _make_data()
        pipe_a = _make_pipeline()
        pipe_b = _make_pipeline()
        out_a = fit_or_get_cached(pipe_a, X, y, cache=None)
        out_b = pipe_b.fit(X, y)
        np.testing.assert_allclose(out_a.predict(X), out_b.predict(X), rtol=0, atol=0)

    def test_first_call_misses_second_call_hits(self):
        """This test ensures the cache misses on first call and hits on second."""
        X, y = _make_data()
        cache = SurrogateCache(maxsize=4)
        pipe1 = _make_pipeline()
        pipe2 = _make_pipeline()
        m1 = fit_or_get_cached(pipe1, X, y, cache=cache)
        m2 = fit_or_get_cached(pipe2, X, y, cache=cache)
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 1
        # Cached result equals the freshly fitted result
        np.testing.assert_allclose(m1.predict(X), m2.predict(X), rtol=0, atol=0)

    def test_different_data_produces_two_misses(self):
        """This test ensures the cache discriminates by training data."""
        X1, y1 = _make_data(seed=1)
        X2, y2 = _make_data(seed=2)
        cache = SurrogateCache(maxsize=4)
        fit_or_get_cached(_make_pipeline(), X1, y1, cache=cache)
        fit_or_get_cached(_make_pipeline(), X2, y2, cache=cache)
        assert cache.stats["misses"] == 2
        assert cache.stats["hits"] == 0

    def test_cached_model_is_independent_from_subsequent_fits(self):
        """This test ensures a stored model is not mutated by future fits.

        The cache does a deep copy on put. After putting, mutating the
        original pipe by re-fitting on different data must not change
        the stored prediction.
        """
        X1, y1 = _make_data(seed=1)
        X2, y2 = _make_data(seed=2)
        cache = SurrogateCache(maxsize=4)
        pipe = _make_pipeline()
        fit_or_get_cached(pipe, X1, y1, cache=cache)
        pred_before = cache.get(fingerprint(X1, y1, pipeline_signature(_make_pipeline()))).predict(X1)
        # Mutate the original pipe with a different dataset
        pipe.fit(X2, y2)
        # Cached entry untouched
        cached = cache.get(fingerprint(X1, y1, pipeline_signature(_make_pipeline())))
        np.testing.assert_allclose(cached.predict(X1), pred_before, rtol=0, atol=0)


# =============================================================================
# End-to-end with EGO
# =============================================================================
@pytest.mark.optimization
class TestEgoWithCache:
    """This class verifies that ego_01_architecture runs identically with cache enabled."""

    def test_ego_with_and_without_cache_match(self):
        """This test ensures cache=SurrogateCache reproduces cache=None on a small benchmark.

        Runs a tiny EGO loop on the sphere benchmark with a fixed seed
        and the SciPy SLSQP inner optimiser. The two runs must produce
        the same OF history because the cache only affects whether
        ``fit`` is recomputed — it never alters the fitted model.
        """
        rng = np.random.default_rng(123)
        d = 2
        n_pop = 6
        n_gen = 3
        x_lower = [-2.0] * d
        x_upper = [2.0] * d
        x_ini = rng.uniform(x_lower, x_upper, size=(n_pop, d)).tolist()

        # Without cache
        _, of_no_cache, _ = ego_01_architecture(
            obj=benchmark.sphere,
            n_gen=n_gen,
            initial_population=x_ini,
            x_lower=x_lower,
            x_upper=x_upper,
            params_opt={"optimizer algorithm": "scipy_slsqp"},
            seed=7,
        )

        # With cache
        cache = SurrogateCache(maxsize=8)
        _, of_with_cache, _ = ego_01_architecture(
            obj=benchmark.sphere,
            n_gen=n_gen,
            initial_population=x_ini,
            x_lower=x_lower,
            x_upper=x_upper,
            params_opt={"optimizer algorithm": "scipy_slsqp"},
            seed=7,
            cache=cache,
        )

        assert of_with_cache == pytest.approx(of_no_cache, rel=1e-12)
        # Each EGO iteration grows the training set by 1 -> n_gen unique fingerprints
        # plus the entry trained inside this run; misses == n_gen.
        assert cache.stats["misses"] == n_gen
        assert cache.stats["hits"] == 0

    def test_second_run_reuses_cache(self):
        """This test ensures running EGO twice with the same seed shares the cache.

        Two consecutive runs with identical seed and initial population
        traverse the same sequence of (X, y) snapshots. The second run
        should therefore produce only cache hits at the fit step.
        """
        rng = np.random.default_rng(123)
        d = 2
        n_pop = 6
        n_gen = 3
        x_lower = [-2.0] * d
        x_upper = [2.0] * d
        x_ini = rng.uniform(x_lower, x_upper, size=(n_pop, d)).tolist()

        cache = SurrogateCache(maxsize=8)
        kwargs = dict(
            obj=benchmark.sphere,
            n_gen=n_gen,
            initial_population=x_ini,
            x_lower=x_lower,
            x_upper=x_upper,
            params_opt={"optimizer algorithm": "scipy_slsqp"},
            seed=7,
            cache=cache,
        )

        ego_01_architecture(**kwargs)
        misses_after_first = cache.stats["misses"]
        ego_01_architecture(**kwargs)
        # Second run hits every fit -> no new misses
        assert cache.stats["misses"] == misses_after_first
        assert cache.stats["hits"] == n_gen
