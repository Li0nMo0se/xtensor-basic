# xtensor-python

You can find a basic usage of xtensor, xtensor-python and xsmid.
The goal is to produce a C++ library importable from python. This library must have at least the same speed as numpy for the vectorized operation.

That speed has been obtained thanks to the [xsmid](https://github.com/xtensor-stack/xsimd) wrapper and optmization compilation flags such as `-O3 -mavx2 -ffast-math`.


## Installation of the dependecies on the system

* See [install](INSTALL.md)

## Build library

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cp mymodule.cpython-38-x86_64-linux-gnu.so ../
```

Note: remember to build in `Release` mode if you want the library to be as fast as Numpy.

## Test library

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 example.py
```