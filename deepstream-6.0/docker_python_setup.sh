#!/bin/bash
apt-get install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt install python3.6 -y
apt install -y libpython3.6
apt remove python3-gi -y
cd /usr/lib/python3/dist-packages/
wget http://mirrors.edge.kernel.org/ubuntu/pool/main/p/pygobject/python3-gi_3.26.1-2ubuntu1_amd64.deb
dpkg -i python3-gi_3.26.1-2ubuntu1_amd64.deb
ln -s /lib/x86_64-linux-gnu/libffi.so.7 /lib/x86_64-linux-gnu/libffi.so.6
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6m 2
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 3
#assume python3.6m is selection 2
echo 2 | update-alternatives --config python3
pip3 install numpy
pip3 install opencv-python
