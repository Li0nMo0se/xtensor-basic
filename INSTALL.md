# Install

Note: Skip the steps if already installed.

## Install python

```
sudo apt install python3.8
```

## Install xtl

```
git clone git@github.com:xtensor-stack/xtl.git
cd xtl/
mkdir build
cd build/
cmake ..
sudo make install -j
```

## Install xtensor

deps:
* xtl

```
git clone git@github.com:xtensor-stack/xtensor.git
cd xtensor/
mkdir build
cd build/
cmake ..
sudo make install -j
```

## Install pybind11

Note: Install pybind globally on the system. it will add files to `/usr/local/include/pybind11` and `/usr/local/share/cmake/pybind11`
```
pip install "pybind11[global]"
```

## Install xtensor-python

deps:
* xtensor
* pybind11

```
git clone git@github.com:xtensor-stack/xtensor-python.git
cd xtensor-python/
mkdir build
cd build/
cmake ..
sudo make install -j
```