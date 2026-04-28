"""Surrogate model cache for the GPR pipeline used by EGO.

The Efficient Global Optimisation loop refits a Gaussian Process
Regressor every iteration. Each fit re-runs the kernel-hyperparameter
optimisation (multi-restart L-BFGS over the marginal log-likelihood),
which is the dominant runtime cost in `ego_01_architecture`. When the
exact same training data and pipeline configuration appear twice —
across replications, across notebook re-runs, across batch experiments
that share an initial population — the second fit produces the same
fitted model, so it should be looked up instead of recomputed.

This module provides a content-addressed cache:

    * `pipeline_signature(pipeline)`           — stable string describing
                                                  every fit-affecting
                                                  hyperparameter
    * `fingerprint(X, y, signature)`           — SHA-256 hex digest used
                                                  as the cache key
    * `SurrogateCache`                         — thread-unsafe in-memory
                                                  LRU + optional disk
                                                  persistence via joblib
    * `fit_or_get_cached(pipe, X, y, cache)`   — drop-in replacement for
                                                  `pipe.fit(X, y)`

Two different kernels, two different `random_state` values, two
different scaler classes — anything that changes the fitted model —
flow into the signature, so cache hits are always semantically
equivalent to a fresh fit. Cached entries are deep-copied on read and
write so subsequent fits cannot mutate stored state.

Resumo em português:
    Cache para o modelo GPR do EGO. A cada iteração do EGO o GPR é
    reajustado; quando os dados de treino e a configuração do pipeline
    são exatamente os mesmos, o ajuste é determinístico — vale guardar.
    A chave é uma impressão digital SHA-256 dos arrays (X, y) somados a
    uma assinatura textual de tudo que influencia o fit (kernel, alpha,
    random_state, etc.). Há armazenamento em memória (LRU) e, opcional,
    em disco via joblib. Cada hit/miss é registrado em ``stats``.
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover  (joblib is in requirements but be resilient)
    joblib = None  # type: ignore

from core.observability import get_logger

_log = get_logger("cache")


__all__ = [
    "SurrogateCache",
    "pipeline_signature",
    "fingerprint",
    "fit_or_get_cached",
]


def _to_2d_float64(data: Any) -> np.ndarray:
    """This helper coerces input arrays to a contiguous float64 2D ndarray.

    Pandas DataFrames, lists of lists and 1D arrays are all accepted.
    The resulting array is what feeds into :func:`fingerprint` so its
    byte representation is deterministic across pandas versions and
    column orders.

    :param data: Input training matrix or target column

    :return: Contiguous float64 ndarray with two dimensions
    """
    if isinstance(data, pd.DataFrame):
        # Use a sorted column order so column-rename without value
        # changes does not invalidate the cache, and the byte layout is
        # deterministic.
        cols = sorted(data.columns.tolist(), key=str)
        arr = data[cols].to_numpy(dtype=np.float64, copy=True)
    elif isinstance(data, pd.Series):
        arr = data.to_numpy(dtype=np.float64, copy=True).reshape(-1, 1)
    else:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
    return np.ascontiguousarray(arr, dtype=np.float64)


def pipeline_signature(pipeline: Any) -> str:
    """This function returns a stable textual signature of a fittable pipeline.

    The signature captures every parameter that changes the fitted
    model: each step's class qualified name, plus all initialisation
    parameters (kernel, alpha, n_restarts_optimizer, random_state,
    normalize_y, scaler with_mean/with_std, etc.). Two pipelines with
    the same signature must yield the same fitted model when fed the
    same (X, y); two pipelines with different signatures may not.

    The implementation uses scikit-learn's ``get_params(deep=True)``,
    which is the canonical way to fully describe an estimator's state.
    The result is sorted by key so the order of parameter introspection
    does not perturb the digest.

    :param pipeline: A fittable estimator (typically ``sklearn.pipeline.Pipeline``)

    :return: Deterministic, human-readable string describing the pipeline
    """
    try:
        params = pipeline.get_params(deep=True)
    except AttributeError:
        params = {"_repr": repr(pipeline)}

    items = []
    for key in sorted(params, key=str):
        value = params[key]
        # Avoid embedding object addresses (e.g. "<RBF at 0x12345>"):
        # use the canonical sklearn repr instead, which is reproducible.
        items.append(f"{key}={value!r}")
    return ";".join(items)


def fingerprint(X: Any, y: Any, signature: str) -> str:
    """This function returns the SHA-256 hex digest used as the cache key.

    The digest depends on three sources: the byte representation of X
    (after coercion to float64 contiguous ndarray with sorted columns),
    the byte representation of y (idem), and the textual pipeline
    signature. SHA-256 is used because it is the standard
    content-addressing hash on the platform — false positives are not
    a practical concern.

    :param X: Training matrix, shape (n, d)
    :param y: Target column, shape (n,) or (n, 1)
    :param signature: Output of :func:`pipeline_signature`

    :return: Hex digest (64 characters)
    """
    x_arr = _to_2d_float64(X)
    y_arr = _to_2d_float64(y)
    h = hashlib.sha256()
    # Tag each section with its shape so two arrays with the same bytes
    # but different shapes (extremely unlikely but possible) cannot
    # collide.
    h.update(f"X{x_arr.shape}".encode("utf-8"))
    h.update(x_arr.tobytes())
    h.update(f"y{y_arr.shape}".encode("utf-8"))
    h.update(y_arr.tobytes())
    h.update(b"|sig|")
    h.update(signature.encode("utf-8"))
    return h.hexdigest()


class SurrogateCache:
    """LRU cache for fitted surrogate-model pipelines.

    The cache stores deep copies so a cached entry is never mutated by
    subsequent calls to ``fit`` on the same pipeline instance. When a
    ``disk_dir`` is provided, every store also writes the model to
    ``disk_dir / <key>.joblib``; on a memory miss the cache transparently
    falls back to the disk and re-populates the in-memory layer.

    :ivar maxsize: Maximum number of entries kept in memory
    :ivar disk_dir: Optional directory for persistent storage; created on demand
    :ivar stats: Dict with ``hits``, ``misses``, ``disk_hits``, ``size``
    """

    def __init__(self, maxsize: int = 128, disk_dir: Optional[Path | str] = None) -> None:
        """This constructor builds an empty surrogate cache.

        :param maxsize: Maximum number of fitted models kept in memory.
                        When exceeded, the least-recently-used entry is
                        evicted from memory (it stays on disk if enabled)
        :param disk_dir: Optional path. When provided, fitted models are
                         persisted as joblib files so the cache survives
                         process restarts. ``None`` keeps the cache
                         in-memory only

        :return: Nothing (instance attributes initialised)
        """
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self.disk_dir = Path(disk_dir) if disk_dir is not None else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._mem: "OrderedDict[str, Any]" = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "disk_hits": 0}

    @property
    def stats(self) -> dict:
        """This property returns a snapshot of cache hit/miss counters.

        :return: Dict with ``hits``, ``misses``, ``disk_hits`` and ``size``
        """
        return {**self._stats, "size": len(self._mem)}

    def __len__(self) -> int:
        """This dunder returns the number of in-memory entries."""
        return len(self._mem)

    def __contains__(self, key: str) -> bool:
        """This dunder reports membership without affecting LRU order."""
        return key in self._mem

    def get(self, key: str) -> Optional[Any]:
        """This method returns a fitted model by key, or None on miss.

        Hits move the entry to the most-recently-used position so the
        next eviction targets a colder key. The returned model is a
        deep copy: the caller can mutate it freely without contaminating
        the cache. Disk-backed caches transparently re-populate memory
        on a memory miss with disk hit.

        :param key: Cache key produced by :func:`fingerprint`

        :return: Deep copy of the cached fitted model, or ``None``
        """
        if key in self._mem:
            self._mem.move_to_end(key)
            self._stats["hits"] += 1
            _log.debug("cache memory hit", extra={"event": "cache.hit",
                                                   "key": key[:16],
                                                   "size": len(self._mem)})
            return copy.deepcopy(self._mem[key])

        if self.disk_dir is not None and joblib is not None:
            disk_path = self.disk_dir / f"{key}.joblib"
            if disk_path.exists():
                model = joblib.load(disk_path)
                # Re-populate memory and bookkeeping
                self._mem[key] = copy.deepcopy(model)
                self._evict_if_needed()
                self._stats["disk_hits"] += 1
                _log.debug("cache disk hit", extra={"event": "cache.disk_hit",
                                                     "key": key[:16],
                                                     "disk_dir": str(self.disk_dir)})
                return model

        self._stats["misses"] += 1
        _log.debug("cache miss", extra={"event": "cache.miss", "key": key[:16]})
        return None

    def put(self, key: str, model: Any) -> None:
        """This method stores a fitted model under the given key.

        A deep copy of the model is kept so future fits on the original
        pipeline instance cannot mutate the cached entry. When a
        ``disk_dir`` is configured, the model is also persisted to disk
        for cross-process reuse.

        :param key: Cache key produced by :func:`fingerprint`
        :param model: Fitted estimator instance (typically a Pipeline)

        :return: Nothing (mutates the cache)
        """
        self._mem[key] = copy.deepcopy(model)
        self._mem.move_to_end(key)
        self._evict_if_needed()
        if self.disk_dir is not None and joblib is not None:
            joblib.dump(model, self.disk_dir / f"{key}.joblib")

    def clear(self) -> None:
        """This method drops every in-memory entry and resets stats.

        Disk-backed entries are kept; the next ``get`` call can
        rehydrate them. Stats are reset so subsequent metrics reflect
        only the cleared epoch.

        :return: Nothing (mutates the cache)
        """
        self._mem.clear()
        self._stats = {"hits": 0, "misses": 0, "disk_hits": 0}

    def _evict_if_needed(self) -> None:
        """This helper enforces the memory size bound via LRU eviction.

        :return: Nothing (mutates ``self._mem``)
        """
        while len(self._mem) > self.maxsize:
            self._mem.popitem(last=False)


def fit_or_get_cached(
    pipeline: Any,
    X: Any,
    y: Any,
    cache: Optional[SurrogateCache],
) -> Any:
    """This function returns a fitted pipeline, consulting the cache first.

    On a hit the cached model is returned without ever calling
    ``fit``. On a miss the function fits the supplied pipeline (which
    is mutated in place by sklearn's contract) and stores a deep copy
    in the cache before returning the fitted instance. Passing
    ``cache=None`` short-circuits the cache and falls back to the
    historical behaviour of plain ``pipeline.fit(X, y)``.

    :param pipeline: An unfitted (or about-to-be-refit) estimator
    :param X: Training matrix
    :param y: Target column
    :param cache: An optional :class:`SurrogateCache`; ``None`` disables caching

    :return: Fitted estimator instance
    """
    if cache is None:
        return pipeline.fit(X, y)

    sig = pipeline_signature(pipeline)
    key = fingerprint(X, y, sig)
    cached = cache.get(key)
    if cached is not None:
        return cached
    fitted = pipeline.fit(X, y)
    cache.put(key, fitted)
    return fitted
