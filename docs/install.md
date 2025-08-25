# Installation

We provide the instructions to install the dependency packages.


## Setup

First, clone the repository locally.

```
git clone https://github.com/buxiangzhiren/HCD
cd HCD
```

Then, install Pytorch, torchvision and the necessary packages as well as pycocotools.
```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt 
pip install 'git+https://github.com/facebookresearch/fvcore' 
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
download weights
```
pip install huggingface_hub
huggingface-cli download --resume-download ali-vilab/text-to-video-ms-1.7b --local-dir ./weight
```

Finally, compile CUDA operators.
```
cd models/ops
python setup.py build install
cd ../..
```
