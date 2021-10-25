import mymodule
import numpy as np
import timeit
import itertools
import joblib
import numba
import pandas as pd
import os

# Part one (xtensor-python)
# Use xsmid to be as fast as numpy
a = np.random.randint(0, 1000, size=100000)
a = a.astype(np.float64)

def sum_of_sines(a):
    return mymodule.sum_of_sines(a)


print("### Xtensor vs Numpy simd operations ###")
sin_python = float(np.sum(np.sin(a)))
sin_cpp = sum_of_sines(a)
assert type(sin_python) == type(sin_cpp)
assert np.isclose(sin_python, sin_cpp).all()
print("Success")

t = timeit.Timer(lambda: sum_of_sines(a))
print(f"C++ version:   {t.timeit(100)}")

t = timeit.Timer(lambda: np.sum(np.sin(a)))
print(f"Numpy version: {t.timeit(100)}")
print()


# Part two (sum axis=2)
## -- Python functions
def local_sum(y, x, res, arr, ksize):
    res[y:y+ksize, x:x+ksize] = np.sum(arr[y:y+ksize, x:x+ksize, :], axis=2)


def python_sum(arr, ksize=10):
    assert arr.ndim == 3

    res = np.empty(shape=arr.shape[:2], dtype=int)

    for y in range(0, arr.shape[0], ksize):
        for x in range(0, arr.shape[1], ksize):
            local_sum(y, x, res, arr, ksize)
    return res

def python_parallel_sum(arr, ksize=10):
    assert arr.ndim == 3

    res = np.empty(shape=arr.shape[:2], dtype=int)

    range_y = range(0, arr.shape[0], ksize)
    range_x = range(0, arr.shape[1], ksize)


    # Do not work with loky, multiprocessing
    mapper = joblib.Parallel(n_jobs=-1, backend="threading")
    mapper(joblib.delayed(local_sum)(y, x, res, arr, ksize) for y in range_y for x in range_x)

    return res

# -- Numba
@numba.jit(nopython=True)
def numba_sum(arr, ksize=10):
    assert arr.ndim == 3

    res = np.empty(shape=arr.shape[:2], dtype=np.int64)

    for y in range(0, arr.shape[0], ksize):
        for x in range(0, arr.shape[1], ksize):
            res[y:y+ksize, x:x+ksize] = np.sum(arr[y:y+ksize, x:x+ksize, :],
                                               axis=2)
    return res

# -- C++ wrapper
def cpp_tbb_sum(arr, ksize=10):
    return mymodule.tbb_sum(arr, ksize)


def cpp_sum(arr, ksize=10):
    return mymodule.sum(arr, ksize)

def cpp_ref_sum(arr):
    return mymodule.ref_sum(arr)

b = np.random.randint(-1000, 1000, size=(2048, 4096, 100), dtype=int)

ref = np.sum(b, axis=2)
ref_cpp = cpp_ref_sum(b)

# Check output, shape and dtype
assert ref.shape == ref_cpp.shape
assert ref.dtype == ref_cpp.dtype
assert np.allclose(ref, ref_cpp)

print("### np.sum(arr, axis=2) ###")
print("Benchmark\n")

nb_it = 2
t = timeit.Timer(lambda: np.sum(b, axis=2))
print(f"Numpy (ref) version:                   {t.timeit(nb_it)}")
t = timeit.Timer(lambda: cpp_ref_sum(b))
print(f"C++ (ref) version:                     {t.timeit(nb_it)}")



print("### double loops execution comparison (with sum axis=2) ###")

ksize = 32
python_res = python_sum(b, ksize=ksize)
python_parallel_res = python_parallel_sum(b, ksize=ksize)
numba_res = numba_sum(b, ksize=ksize)
cpp_tbb_res = cpp_tbb_sum(b, ksize=ksize)
cpp_res = cpp_sum(b, ksize=ksize)

# Check output, shape and dtype
assert ref.shape == python_res.shape
assert ref.dtype == python_res.dtype
assert np.isclose(ref, python_res).all()

assert ref.shape == python_parallel_res.shape
assert ref.dtype == python_parallel_res.dtype
assert np.isclose(ref, python_parallel_res).all()

assert ref.shape == numba_res.shape
assert ref.dtype == numba_res.dtype
assert np.isclose(ref, numba_res).all()

assert ref.shape == cpp_res.shape
assert ref.dtype == cpp_res.dtype
assert np.isclose(ref, cpp_res).all()

assert ref.shape == cpp_tbb_res.shape
assert ref.dtype == cpp_tbb_res.dtype
assert np.isclose(ref, cpp_tbb_res).all()
print("Success")

print("\nBenchmark")
nb_it = 2
# Check timing
print(f"Number of iteration: {nb_it}")
print(f"Input shape: {b.shape}")
print(f"Input dtype: {b.dtype}, Output dtype: {ref.dtype}")

ksizes = [10, 25, 32, 64, 128]
times = []

for ksize in ksizes:
    time = []

    print(f"\nksize {ksize}")

    t = timeit.Timer(lambda: python_sum(b, ksize=ksize))
    time.append(t.timeit(nb_it))
    print(f"Python (without joblib) version:       {time[-1]}")
    t = timeit.Timer(lambda: python_parallel_sum(b, ksize=ksize))
    time.append(t.timeit(nb_it))
    print(f"Python (with joblib) version:          {time[-1]}")


    t = timeit.Timer(lambda: numba_sum(b, ksize=ksize))
    time.append(t.timeit(nb_it))
    print(f"Numba version:                         {time[-1]}")


    t = timeit.Timer(lambda: cpp_tbb_sum(b, ksize=ksize))
    time.append(t.timeit(nb_it))
    print(f"C++ (with tbb) version:                {time[-1]}")
    t = timeit.Timer(lambda: cpp_sum(b, ksize=ksize))
    time.append(t.timeit(nb_it))
    print(f"C++ (without tbb) version:             {time[-1]}")

    times.append(time)


columns = ["Python (without joblib) version",
           "Python (with joblib) version",
           "Numba version",
           "C++ (with tbb) version",
           "C++ (without tbb) version"]
df = pd.DataFrame(times, index=ksizes, columns=columns)
filename = f"benchmark{b.shape}.csv"
if os.path.exists(filename):
    os.remove(filename)
df.to_csv(filename)
print(f"Benchmark saved {filename}")