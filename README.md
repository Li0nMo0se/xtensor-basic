# xtensor-python

You can find a basic usage of xtensor, xtensor-python and xsmid.
The goal is to produce a C++ library importable from python. This library must have at least the same speed as numpy for the vectorized operation.

The xtensor speed has been obtained thanks to:
* the [xsmid](https://github.com/xtensor-stack/xsimd) wrapper
* Optmization compilation flags such as `-O3 -mavx2 -ffast-math`.
* the [tbb](https://github.com/oneapi-src/oneTBB) parallelization library

## Program

### Xtensor vs Numpy simd operations

Compare the speed of simd operation with numpy and xtensor

#### Numpy

```Python
np.sum(np.sin(a))
```

#### C++/Xtensor

```c++
double sum_of_sines(xt::pyarray<double>& m)
{
    return xt::sum(xt::sin(m), {0})(0);
}
```

#### Benchmark

```
Numpy version: 0.02018828600012057
C++ version:   0.019397275000301306
```

The two operation took about the same processing time. Xtensor clearly uses smid instructions to perform as fast as numpy

### Double loops execution comparison (with sum axis=2)

Compare the speed of execution double loop with numpy (parallel or not) and xtensor (parallel or not). The goal is to simulate python loop.

Note: The example simulates the behavior of np.sum(axis=2). There are loops for the sake of the benchmarks.

#### Basic Python code (tagged as Python without joblib)

```Python
def python_sum(arr, ksize=10):
    assert arr.shape[0] % ksize == 0
    assert arr.shape[1] % ksize == 0
    assert arr.ndim == 3

    res = np.empty(shape=arr.shape[:2])

    for y in range(0, arr.shape[0], ksize):
        for x in range(0, arr.shape[1], ksize):
    	    res[y:y+ksize, x:x+ksize] = np.sum(arr[y:y+ksize, x:x+ksize, :], axis=2)
    return res
```

#### Python (with joblib)

```Python
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
```

Note that here, we want to use the threading backend with as many jobs as possible.

#### C++ (without tbb)

```c++
xt::pyarray<int> sum(xt::pyarray<int>& m, unsigned int ksize)
{
    assert(m.dimension() == 3);
    assert(m.shape(0) % ksize == 0);
    assert(m.shape(1) % ksize == 0);
    xt::xarray<int>::shape_type shape = {m.shape(0), m.shape(1)};
    xt::xarray<int> res(shape);

    // auto range_y = xt::range(0, m.shape(0), ksize);
    // auto range_x = xt::range(0, m.shape(1), ksize);

    for (size_t y = 0; y != m.shape(0); y += ksize)
    {
        for (size_t x = 0; x != m.shape(1); x += ksize)
        {
            auto m_view = xt::view(m,
                                   xt::range(y, y + ksize),
                                   xt::range(x, x + ksize),
                                   xt::all());
            xt::view(res, xt::range(y, y + ksize), xt::range(x, x + ksize)) =
                xt::sum(m_view, {2});
        }
    }
    return res;
}
```

Note that the code look very much like Numpy.

#### C++ (with tbb)

```c++
xt::pyarray<int> tbb_sum(xt::pyarray<int>& m, unsigned int ksize)
{
    assert(m.dimension() == 3);
    assert(m.shape(0) % ksize == 0);
    assert(m.shape(1) % ksize == 0);

    xt::xarray<int>::shape_type shape = {m.shape(0), m.shape(1)};
    xt::xarray<int> res(shape);

    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, m.shape(0) / ksize, 0, m.shape(1) / ksize),
        [&res, &m, ksize](const tbb::blocked_range2d<int>& r) {
            for (int y = std::begin(r.rows()); y != std::end(r.rows()); y++)
            {
                for (int x = std::begin(r.cols()); x != std::end(r.cols()); x++)
                {
                    size_t real_y = y * ksize;
                    size_t real_x = x * ksize;
                    auto m_view = xt::view(m,
                                           xt::range(real_y, real_y + ksize),
                                           xt::range(real_x, real_x + ksize),
                                           xt::all());
                    xt::view(res,
                             xt::range(real_y, real_y + ksize),
                             xt::range(real_x, real_x + ksize)) =
                        xt::sum(m_view, {2});
                }
            }
        });
    return res;
}
```

In the tbb approach, the goal is to use extensively **threads** and xtensor **smid** instructiosn within every threads.

#### Benchmarks

![](benchmark/benchmark(2000,%203000,%20100)_bar.png)

For ksize=1, it can be seen that the Python joblib threading approach is much slower. This is because of the overhead using threads. On the other hand, the xtensor with tbb is the fastest.

Let's zoom in for ksize=[10, 100, 1000]

![](benchmark/benchmark(2000,%203000,%20100)_cut_bar.png)

* Python (without joblib): get faster with greater ksize. This can be explained because greater the ksize is, less iterations we do. **More simd instructions and less iterations give speed**.
* Python (with joblib): for ksize=10, it is still the slowest. But from ksize=100 it becomes the fastest. That means more smid instructions and less iterations combined with a reasonable amount of threads is **the fastest**.
* Numba: get slower for greater ksize. What would be the reason? It should be at least as fast as regular python loop.
* C++ (without tbb): The speed is constant and slower than regular python loop. This is **the disapointing part**. I was expecting more speed than regular python loop as both uses smid instruction but c++ is moreover compiled.
* C++ (with tbb): for small ksize, this is the fastest options. However, from ksize=100, even with tbb threads, it is slower than regular python loops and numpy smid instruction?

Let's see more ksizes $\in [10-100]$

![](benchmark/benchmark(2000,%203000,%20100)_bar_manyksizes.png)

**Recap:**

Best for ksize:
* 1 -> xtensor with tbb
* 10 -> xtensor with tbb
* 25 -> python
* 32 -> python
* 64 -> python joblib
* 100 -> python joblib
* 1000 -> python joblib

For small ksize, it is better to use xtensor with tbb on that example. For greater ksize, joblib gives the best speed. In between, regular python is the fastest.

**I believe the best is to use as many smid instructions as possible and use some threads performing those big smid instructions.**

![](benchmark/benchmark(2000,%203000,%20100)_cut_plot.png)

Finally, from ksize=100 and ksize=1000 the speed for each approach is the same but Numba. **Therefore, a ksize greater than 100 do not improve the speed**.
#### Questions:

**What is our target ksize?**
## Build library

### Installation of the dependecies on the system

* See [install](INSTALL.md)

### Cmake build

In the root directory of the project:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cp mymodule.cpython-38-x86_64-linux-gnu.so ../
```

Note: remember to build in `Release` mode if you want the library to be as fast as Numpy.

## Test & benchmark

In the build directory you will find a `run.sh` script that will run the compilation and then the benchmark scripts `example.py` in a python venv


### Execute the benchmark only

Alternatively, to execute the benchmark, you may do the following in the root directory:
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 example.py
```

### Plot the benchmark

For the basic plot:
```
python3 plot.py --filename BENCHMARK.csv
```

See the `help` section of `plot.py` for more options
```
python3 plot.py --help
```