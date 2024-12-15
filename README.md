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

## Notes:
- Here i am using 'buffalo_l' model to detect faces in image and inswapper_128.onnx model to swap the image faces.<br>
- You can contol swap the image faces using source_indexes and target_indexes.<br>
- For example if source and target image have many face then source_indexes = 0,1 and target_indexes=1,3 (indesxes starting from 0) will change faces of target image at 1,3 index from source index 0,1. <br>

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

you will see interface like this

<left><img src="https://github.com/yashkothadia/Face_swap/blob/main/face_swap/face_swap.png" width="49%" height="49%"></left>

You will obtain the exact result as above.

