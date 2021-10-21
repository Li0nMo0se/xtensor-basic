# xtensor-python

## Build library

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cp mymodule.cpython-38-x86_64-linux-gnu.so ../
```

## Test library

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 example.py
```

## Installation of the dependecies on the system

* See [install](INSTALL.md)