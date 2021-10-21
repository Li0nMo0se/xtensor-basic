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
git checkout 0.7.0
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
git checkout 0.24.0
mkdir build
cd build/
cmake ..
sudo make install -j
```

## Install xsimd

```
git clone git@github.com:xtensor-stack/xsimd.git
cd xsimd/
git checkout 8.0.3
mkdir build
cd build
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

## Install tbb

**Note:** I did not find a proper way to install tbb with cmake support

Download tbb from [tbb release](https://github.com/oneapi-src/oneTBB/releases) and pick the linux release. The most recent is `oneapi-tbb-2021.4.0-lin.tgz` (when the file was written)

Where the archive was downloaded, do the following:
```
tar -xvf oneapi-tbb-2021.4.0-lin.tgz
sudo mv env/ /usr/local/
sudo mv include/* /usr/local/include/
sudo mv lib/* /usr/local/lib/
sudo mv lib/cmake/* /usr/local/lib/cmake/
sudo mv lib/pkgconfig/* /usr/local/lib/pkgconfig/
echo "source /usr/local/env/vars.sh" >> ~/.bashrc
sudo mv /usr/local/lib/cmake/tbb/TBBConfig.cmake /usr/local/lib/cmake/tbb/tbbConfig.cmake
```