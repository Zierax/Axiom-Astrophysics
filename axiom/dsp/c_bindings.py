import ctypes
import logging
import os
import sys

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure Python Fallbacks for Complexity Metrics
# ---------------------------------------------------------------------------

def py_shannon_entropy_norm(signal, bins=128):
    if len(signal) < 2:
        return 0.0
    min_val, max_val = np.min(signal), np.max(signal)
    if max_val == min_val:
        return 0.0
    counts, _ = np.histogram(signal, bins=bins, range=(min_val, max_val))
    counts = counts[counts > 0]
    probs = counts / len(signal)
    entropy = -np.sum(probs * np.log2(probs))
    return float(np.clip(entropy / np.log2(bins), 0.0, 1.0))

def py_permutation_entropy(signal, order=3, delay=1):
    n = len(signal)
    if n < order * delay:
        return 0.0
    
    # Generate patterns
    patterns = []
    for i in range(n - (order - 1) * delay):
        pattern = [signal[i + j * delay] for j in range(order)]
        patterns.append(tuple(np.argsort(pattern)))
        
    unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / len(patterns)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(np.math.factorial(order))
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))

def py_higuchi_fractal_dimension(signal, kmax=10):
    n = len(signal)
    if n < 20:
        return 1.5
    
    L = []
    for k in range(1, kmax + 1):
        Lk = 0.0
        for m in range(k):
            # Calculate length of curve
            indices = np.arange(m, n - k, k)
            if len(indices) == 0:
                continue
            diffs = np.abs(signal[indices + k] - signal[indices])
            Lmk = np.sum(diffs) * (n - 1) / (len(indices) * k * k)
            Lk += Lmk
        L.append(Lk / k)
        
    # Fit line: log(L) vs log(1/k)
    x = np.log(1.0 / np.arange(1, kmax + 1))
    y = np.log(np.array(L) + 1e-12)
    slope, _ = np.polyfit(x, y, 1)
    return float(np.clip(slope, 1.0, 2.0))

def py_lz76_complexity(signal):
    n = len(signal)
    if n < 2:
        return 0.0
    
    # Binarize using median
    median = np.median(signal)
    s = (signal > median).astype(int)
    
    # LZ76 algorithm
    complexity = 1
    l = 1
    k = 1
    
    while l + k <= n:
        # Check if s[l:l+k] is a substring of s[0:l+k-1]
        # s_sub is s[l:l+k], s_history is s[0:l+k-2]
        match = False
        sub = s[l-1 : l+k-1]
        hist = s[0 : l+k-2]
        
        # Simple substring match
        for idx in range(len(hist) - len(sub) + 1):
            if np.array_equal(hist[idx : idx + len(sub)], sub):
                match = True
                break
                
        if not match:
            complexity += 1
            l += k
            k = 1
        else:
            k += 1
            
        if l + k > n:
            complexity += 1
            break
            
    norm = n / (np.log2(n) + 1.0)
    return float(np.clip(complexity / norm, 0.0, 1.0))

# ---------------------------------------------------------------------------
# C Core Shared Library Binding Loader
# ---------------------------------------------------------------------------

_c_lib = None
_loaded_path = None

def load_c_library():
    global _c_lib, _loaded_path
    if _c_lib is not None:
        return _c_lib
        
    # Find package directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Possible paths for the compiled library
    library_names = []
    if sys.platform == "win32":
        library_names = ["axiom_core.dll", "libaxiom_core.dll"]
    elif sys.platform == "darwin":
        library_names = ["axiom_core.dylib", "libaxiom_core.dylib"]
    else:
        library_names = ["axiom_core.so", "libaxiom_core.so"]
        
    search_dirs = [
        os.path.join(base_dir, "lib"),
        os.path.join(base_dir, "Axiom_C"),
        base_dir,
        "."
    ]
    
    for search_dir in search_dirs:
        for lib_name in library_names:
            path = os.path.join(search_dir, lib_name)
            if os.path.exists(path):
                try:
                    lib = ctypes.CDLL(path)
                    
                    # Define argument & return types for ctypes functions
                    lib.shannon_entropy_norm.argtypes = [
                        ctypes.POINTER(ctypes.c_double), 
                        ctypes.c_int, 
                        ctypes.c_int
                    ]
                    lib.shannon_entropy_norm.restype = ctypes.c_double
                    
                    lib.permutation_entropy.argtypes = [
                        ctypes.POINTER(ctypes.c_double), 
                        ctypes.c_int, 
                        ctypes.c_int, 
                        ctypes.c_int
                    ]
                    lib.permutation_entropy.restype = ctypes.c_double
                    
                    lib.higuchi_fractal_dimension.argtypes = [
                        ctypes.POINTER(ctypes.c_double), 
                        ctypes.c_int, 
                        ctypes.c_int
                    ]
                    lib.higuchi_fractal_dimension.restype = ctypes.c_double
                    
                    lib.lz76_complexity.argtypes = [
                        ctypes.POINTER(ctypes.c_double), 
                        ctypes.c_int
                    ]
                    lib.lz76_complexity.restype = ctypes.c_double
                    
                    _c_lib = lib
                    _loaded_path = path
                    return _c_lib
                except Exception as e:
                    log.warning("Failed to load C library at %s: %s", path, e)
                    
    # Return None if not found
    return None

# Attempt to load library immediately
load_c_library()

# ---------------------------------------------------------------------------
# Unified Public Interface (Invokes C-functions if loaded, else fallbacks to Python)
# ---------------------------------------------------------------------------

def shannon_entropy_norm(signal, bins=128):
    lib = load_c_library()
    if lib is not None:
        try:
            arr = np.ascontiguousarray(signal, dtype=np.float64)
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return float(lib.shannon_entropy_norm(ptr, len(arr), bins))
        except Exception as exc:
            log.debug("C library call failed, using pure-Python fallback: %s", exc)
    return py_shannon_entropy_norm(signal, bins)

def permutation_entropy(signal, order=3, delay=1):
    lib = load_c_library()
    if lib is not None:
        try:
            arr = np.ascontiguousarray(signal, dtype=np.float64)
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return float(lib.permutation_entropy(ptr, len(arr), order, delay))
        except Exception as exc:
            log.debug("C library call failed, using pure-Python fallback: %s", exc)
    return py_permutation_entropy(signal, order, delay)

def higuchi_fractal_dimension(signal, kmax=10):
    lib = load_c_library()
    if lib is not None:
        try:
            arr = np.ascontiguousarray(signal, dtype=np.float64)
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return float(lib.higuchi_fractal_dimension(ptr, len(arr), kmax))
        except Exception as exc:
            log.debug("C library call failed, using pure-Python fallback: %s", exc)
    return py_higuchi_fractal_dimension(signal, kmax)

def lz76_complexity(signal):
    lib = load_c_library()
    if lib is not None:
        try:
            arr = np.ascontiguousarray(signal, dtype=np.float64)
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return float(lib.lz76_complexity(ptr, len(arr)))
        except Exception as exc:
            log.debug("C library call failed, using pure-Python fallback: %s", exc)
    return py_lz76_complexity(signal)
