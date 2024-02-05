# The Repo contains the files and commands needed to install TensorFlow, PyTorch and ONNXRuntime on RPi

# Initial Command
````
sudo apt-get update
sudo apt-get upgrade
````
# Install Numpy
````
pip3 install numpy
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
sudo apt-get install wget
wget https://files.pythonhosted.org/packages/36/60/aa7bf18070611e7b019886d34516337ce6a2fe9da60745bc90b448642a10/torch-2.0.0-cp39-cp39-manylinux2014_aarch64.whl
pip3 install torch-2.0.0-cp39-cp39-manylinux2014_aarch64.whl
pip3 install https://files.pythonhosted.org/packages/89/12/14e3114ac16ab113b984ddb22116773a79e5b5b7c10de5d21676e03f3abc/torchvision-0.15.1-cp39-cp39-manylinux2014_aarch64.whl
pip3 install pandas
pip3 install PyYAML
pip3 install matplotlib
pip3 install seaborn
pip3 install scipy
````

# Install MediaPipie For Raspbian Buster
````
sudo apt install ffmpeg -y
sudo apt install libxcb-shm0 libcdio-paranoia-dev libsdl2-2.0-0 libxv1  libtheora0 libva-drm2 libva-x11-2 libvdpau1 libharfbuzz0b libbluray2 libatlas-base-dev libhdf5-103 libgtk-3-0 libdc1394-22 libopenexr23 -y
pip3 install mediapipe-rpi3
````

# Install MediaPipie For Raspbian Bullseye
````
sudo apt install ffmpeg -y
sudo apt install libxcb-shm0 libcdio-paranoia-dev libsdl2-2.0-0 libxv1  libtheora0 libva-drm2 libva-x11-2 libvdpau1 libharfbuzz0b libbluray2 libatlas-base-dev libhdf5-103 libgtk-3-0 libdc1394-22 libopenexr23 -y
pip3 install mediapipe --user
````

# Install ONNXRuntime
Clone the git repository and run,
````
pip3 install onnxruntime-1.14.1-cp39-cp39-manylinux_2_27_aarch64.whl
````

# Install Matplotlib
Clone the git repository and run,
````
pip3 install matplotlib-3.8.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
````
Or
```
sudo apt-get install python3-matplotlib
```

Any other libraries you want for your implementation should be possible to just pip install it
