# Face swap

## Installation

```bash
# git clone this repository
git clone https://github.com/yashkothadia/Face_swap.git
cd Face_swap/face_swap/

# create a Python venv
python3 -m venv venv

# activate the venv
source venv/bin/activate

# install required packages
pip install -r requirements.txt
```

You have to install ``onnxruntime-gpu`` manually to enable GPU inference, install ``onnxruntime`` by default to use CPU only inference.

## Download Checkpoints

First, you need to download [face swap model](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx) and save it under `Faceswap/checkpoints`

```bash
cd Faceswap
mkdir checkpoints
wget -O ./checkpoints/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
```


## Quick Inference

```bash
# run django server
pythom manage.py runserver
```

you will see interface like this (https://github.com/yashkothadia/Face_swap/blob/main/face_swap/face_swap.png)


<left><img src="https://github.com/yashkothadia/Face_swap/blob/main/face_swap/face_swap.png" width="49%" height="49%"></left> 

You will obtain the exact result as above.

