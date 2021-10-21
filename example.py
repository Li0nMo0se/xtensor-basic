import mymodule
import numpy as np
import timeit
import itertools
import joblib
import numba

# Part one (xtensor-python)
# Use xsmid to be as fast as numpy
a = np.random.randint(0, 1000, size=10000)

def sum_of_sines(a):
    return mymodule.sum_of_sines(a)


print("### Xtensor vs Numpy simd operations ###")
assert np.isclose(np.sum(np.sin(a)), sum_of_sines(a)).all()
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
    assert arr.shape[0] % ksize == 0
    assert arr.shape[1] % ksize == 0
    assert arr.ndim == 3

    res = np.empty(shape=arr.shape[:2])

    for y in range(0, arr.shape[0], ksize):
        for x in range(0, arr.shape[1], ksize):
            local_sum(y, x, res, arr, ksize)
    return res

def python_parallel_sum(arr, ksize=10):
    assert arr.shape[0] % ksize == 0
    assert arr.shape[1] % ksize == 0
    assert arr.ndim == 3

    res = np.empty(shape=arr.shape[:2])

    range_y = range(0, arr.shape[0], ksize)
    range_x = range(0, arr.shape[1], ksize)


    # Do not work with loky, multiprocessing
    mapper = joblib.Parallel(n_jobs=-1, backend="threading")
    mapper(joblib.delayed(local_sum)(y, x, res, arr, ksize) for y in range_y for x in range_x)

    return res

# -- Numba
@numba.jit(nopython=True)
def numba_sum(arr, ksize=10):
    assert arr.shape[0] % ksize == 0
    assert arr.shape[1] % ksize == 0
    assert arr.ndim == 3

    res = np.empty(shape=arr.shape[:2])

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

b = np.random.randint(-1000, 1000, size=(1300, 2000, 100))

ref = np.sum(b, axis=2)

ksize = 10
python_res = python_sum(b, ksize=ksize)
python_parallel_res = python_parallel_sum(b, ksize=ksize)
numba_res = numba_sum(b, ksize=ksize)
cpp_res = cpp_sum(b, ksize=ksize)
cpp_tbb_res = cpp_tbb_sum(b, ksize=ksize)

print("### double loops execution comparison (with sum axis=2) ###")
assert np.isclose(ref, python_res).all()
assert np.isclose(ref, python_parallel_res).all()
assert np.isclose(ref, numba_res).all()
assert np.isclose(ref, cpp_res).all()
assert np.isclose(ref, cpp_tbb_res).all()
print("Success")

print("\nBenchmark")
nb_it = 2
# Check timing
print(f"Number of iteration: {nb_it}")
print(f"Input shape: {b.shape}")

for ksize in (1, 10, 100):
    print(f"\nksize {ksize}")
    t = timeit.Timer(lambda: np.sum(b, axis=2))
    print(f"Numpy version (no loop, not relevant): {t.timeit(nb_it)}")

    t = timeit.Timer(lambda: python_sum(b, ksize=ksize))
    print(f"Python (without joblib) version:       {t.timeit(nb_it)}")
    t = timeit.Timer(lambda: python_parallel_sum(b, ksize=ksize))
    print(f"Python (with joblib) version:          {t.timeit(nb_it)}")


    t = timeit.Timer(lambda: numba_sum(b, ksize=ksize))
    print(f"Numba version:                         {t.timeit(nb_it)}")

    t = timeit.Timer(lambda: cpp_tbb_sum(b, ksize=ksize))
    print(f"C++ (with tbb) version:                {t.timeit(nb_it)}")
    t = timeit.Timer(lambda: cpp_sum(b, ksize=ksize))
    print(f"C++ (without tbb) version:             {t.timeit(nb_it)}")



