# The Repo contains the files and commands needed to install TensorFlow, PyTorch and ONNXRuntime on RPi

# Initial Command
````
sudo apt-get update
sudo apt-get upgrade
````

# Install TensorFlow
````
sudo -H pip3 install --upgrade protobuf==3.20.0

git clone -b v0.23.1 --depth=1 --recursive https://github.com/tensorflow/io.git
cd io
python3 setup.py -q bdist_wheel --project tensorflow_io_gcs_filesystem
cd dist
sudo -H pip3 install tensorflow_io_gcs_filesystem-0.23.1-cp39-cp39-linux_aarch64.whl
cd ~

sudo -H pip3 install gdown
gdown https://drive.google.com/uc?id=1G2P-FaHAXJ-UuQAQn_0SYjNwBu0aShpd
sudo -H pip3 install tensorflow-2.10.0-cp39-cp39-linux_aarch64.whl
````

# Install OpenCV only for Python
````
sudo apt-get install python3-opencv -y
````

# Install TFLite
````
sudo apt-get install libopenblas-dev -y
sudo apt-get install libcblas-dev -y
sudo apt-get install libhdf5-dev  -y
sudo apt-get install libhdf5-serial-dev -y
sudo apt-get install libatlas-base-dev -y
sudo apt-get install libjasper-dev  -y
sudo apt-get install libqtgui4  -y
sudo apt-get install libqt4-testv -y
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime
````


# Install Torch
Clone the git repository and run,
````
pip3 install torch-1.9.0a0+gitd69c22d-cp39-cp39-linux_aarch64.whl
````

# Install ONNXRuntime
````
pip3 install onnxruntime-1.14.1-cp39-cp39-manylinux_2_27_aarch64.whl
````

Any other libraries you want for your implementation should be possible to just pip install it
