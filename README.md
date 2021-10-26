# xtensor-python

You can find a basic usage of xtensor, xtensor-python and xsmid.
The goal is to produce a C++ library importable from python. This library must have at least the same speed as numpy for the vectorized operation.

The xtensor speed has been obtained thanks to:
* the [xsmid](https://github.com/xtensor-stack/xsimd) wrapper
* Optmization compilation flags such as `-O3 -mavx2 -ffast-math`.
* the [tbb](https://github.com/oneapi-src/oneTBB) parallelization library

While coding a xtensor function binded to python, **be aware of the type of arrays to avoid slow casting** especially while declaring xt::pyarray from a numpy array to a xtensor array and the way back

## Program

### 1. Xtensor vs Numpy simd operations (`np.sum(np.sin(a))`)

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


Test data:
```python
a = np.random.randint(0, 1000, size=100000)
a = a.astype(np.float64) # double
```

Times:
```
C++ version:   0.17637319000095886
Numpy version: 0.18001604199889698
```

The two operation took about the same processing time. Xtensor clearly uses smid instructions to perform as fast as numpy

### 2. Xtensor vs Numpy simd operations (`np.sum(a, axis=2)`)


#### Numpy

```Python
ref = np.sum(b, axis=2)
```

#### C++/Xtensor

```C++
xt::pyarray<int64_t> ref_sum(xt::pyarray<int64_t>& m)
{
    return xt::sum<int64_t>(m, {2}, xt::evaluation_strategy::immediate);
}
```
#### Benchmark

Test data:
```Python
b = np.random.randint(-1000, 1000, size=(2048, 4096, 100), dtype=int) # int64
```

Times:
```
Numpy (ref) version:                   1.08570130499902
C++ (ref) version:                     0.8704822669988062
```

`Xtensor` is slightly faster than `Python`!

### 3. Double loops execution comparison (`np.sum(a, axis=2)` with loops)

Compare the speed of execution double loop with numpy (parallel or not), xtensor (parallel or not) and numba. The goal is to simulate python loops.

Note: The example simulates the behavior of np.sum(a, axis=2). There are loops for the sake of the benchmarks.

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
def local_sum(y, x, res, arr, ksize):
    res[y:y+ksize, x:x+ksize] = np.sum(arr[y:y+ksize, x:x+ksize, :], axis=2)

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

```C++
xt::pyarray<int64_t> sum(xt::pyarray<int64_t>& m, unsigned int ksize)
{
    assert(m.dimension() == 3);
    xt::xarray<int64_t>::shape_type shape = {m.shape(0), m.shape(1)};
    xt::xarray<int64_t> res(shape);

    // auto range_y = xt::range(0, m.shape(0), ksize);
    // auto range_x = xt::range(0, m.shape(1), ksize);

    for (size_t y = 0; y < m.shape(0); y += ksize)
    {
        for (size_t x = 0; x < m.shape(1); x += ksize)
        {
            auto m_view = xt::view(m,
                                   xt::range(y, y + ksize),
                                   xt::range(x, x + ksize),
                                   xt::all());
            // std::cout << xt::adapt(m_view.shape()) << "\n";
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
xt::pyarray<int64_t> tbb_sum(xt::pyarray<int64_t>& m, unsigned int ksize)
{
    assert(m.dimension() == 3);

    xt::xarray<int64_t>::shape_type shape = {m.shape(0), m.shape(1)};
    xt::xarray<int64_t> res(shape);

    tbb::parallel_for(
        tbb::blocked_range2d<int>(
            0,
            m.shape(0) / ksize + (m.shape(0) % ksize != 0),
            0,
            m.shape(1) / ksize + (m.shape(1) % ksize != 0)),
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

In the tbb approach, the goal is to use extensively **threads** and xtensor **smid** instructions within every threads.

#### Benchmarks

```
Benchmark
Number of iteration: 2
Input shape: (2048, 4096, 100)
Input dtype: int64, Output dtype: int64

ksize 10
Python (without joblib) version:       2.975913121999838
Python (with joblib) version:          22.099191903000246
Numba version:                         4.010191121000389
C++ (with tbb) version:                0.7508320460001414
C++ (without tbb) version:             2.2969875620001403

ksize 25
Python (without joblib) version:       1.4691777710013412
Python (with joblib) version:          3.650076881000132
Numba version:                         5.691073715999664
C++ (with tbb) version:                0.7174011890001566
C++ (without tbb) version:             2.2594705099982093

ksize 32
Python (without joblib) version:       1.3399212210006226
Python (with joblib) version:          2.213738635999107
Numba version:                         5.6206413060008344
C++ (with tbb) version:                0.7110747160004394
C++ (without tbb) version:             2.2406729869999253

ksize 64
Python (without joblib) version:       1.16353083300055
Python (with joblib) version:          0.7941494320002676
Numba version:                         8.846110742000747
C++ (with tbb) version:                0.708892458000264
C++ (without tbb) version:             2.1846884330007015

ksize 128
Python (without joblib) version:       1.1405782170004386
Python (with joblib) version:          0.6862048050006706
Numba version:                         9.91710225299903
C++ (with tbb) version:                0.7053286730006221
C++ (without tbb) version:             2.174009702999683
```

![](benchmark/benchmark(2048,%204096,%20100)_bar.png)

##### ksize=10

It can be seen that the Python joblib threading approach is much slower. This is because of the overhead using threads. On the other hand, the xtensor and python approach with simple loops are almost as fast. On the other hand, xtensor with tbb is the fastest (**3.9 times faster than classic python loops**).

##### ksize=25

Same as before, but joblib would use less threads which enforces a lower threads overheads. This leads to the joblib approach being faster but still slower than python classic loops. On the other hand, numba became the slowest.

The xtensor without tbb approach becomes slower than the classic python loops approach. But the xtensor with tbb is still the fastest (**2 times faster than the classic python loops**)

##### ksize=32

Joblib approach becomes faster again because there are less threads involved.


##### ksize=64

Numba is getting much slower. The joblib approach is finally faster than the classic python loop (but just a bit faster). The xtensor without tbb did not improve (why?). The xtensor with tbb is still the fastest, slightly faster than the joblib approach.

##### ksize=128

Numba is the slowest by far. Xtensor without tbb is the second slowest. The two parallel approaches (with joblib and xtensor) are as fast

##### Recap

| ksize | Fastest approach | How fast compared to classic python loops |
| -----   | ----------- | ------ |
| 10   | xtensor with tbb | x3.9 |
| 25   | xtensor with tbb | x2 |
| 32  | xtensor with tbb | x1.8 |
| 64  | xtensor with tbb  | x1.64 |
| 128 | xtensor with tbb and joblib  | x1.61 |

It is normal to not have a big improvements for large ksize because most of the operations will be perfomed with smid intructions so parallelizing is not that required.

**I believe the best is to use as many smid instructions as possible and use some threads performing those big smid instructions (all compiled in C++)**

**Moreover, in every cases, the xtensor with tbb approach seems the fastest**.

#### Questions:

What is our target ksize?
## Build library

### Installation of the dependecies on the system

* See [installation guide](INSTALL.md)

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