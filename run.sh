INVENV=$(python3 -c 'import sys; print ("1" if hasattr(sys, "real_prefix") else "0")')
if [ $INVENV -eq 0 ]; then
    cd .. && python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt && cd -
fi
make && cp mymodule.cpython-38-x86_64-linux-gnu.so .. && cd .. && python3 example.py
