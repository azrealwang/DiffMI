# Breaking Face Recognition Privacy via Diffusion-Driven Training-Free Model Inversion

This is a demo of our proposed model inversion attack, **DiffMI**.

****
## Contents
* [Main Requirements](#Main-Requirements)
* [Installation](#Installation)
* [Target Faces](#Target-Faces)
* [Target Models](#Target-Models)
* [Usage](#Usage)
* [User Study](#User-Study)

****

## Main Requirements

  * **Python (3.9.23)**
  * **torch (2.1.2+cu118)**
  * **torchvision (0.16.2+cu118)**
  * **torchjpeg ()**
  * **facenet-pytorch (2.5.3)**
  * **diffusers ()**
  * **accelerate (0.26.1)**
  * **scipy ()**
  * **scikit-learn ()**
  
  The versions in `()` have been tested.

## Installation
```
git clone https://github.com/xxxx/DiffMI.git
cd DiffMI
```

if equipped with GPU:
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install --no-deps torchjpeg
pip install -r requirements.txt
```

otherwise:
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install --no-deps torchjpeg
pip install -r requirements.txt
```

## Target Faces

The image name must satisfy `00000_0.jpg`. `00000` and `_0` indicates the image id and user id/class/label, respectively. The image id must be unique and auto-increment from `00000`. `.jpg` can be any image file format.

20 target images have been prepared in `imgs/target/` for running [demos](#Usage).

## Target Models

* [FaceNet](https://github.com/timesler/facenet-pytorch): InceptionResNetV1 pretrained using the VGGFACE2 dataset; automatically downloaded.

* [ArcFace](https://github.com/deepinsight/insightface): IResNet100 pretrained using the MS1MV2 dataset; automatically downloaded.

* [DCTDP](https://github.com/Tencent/TFace/tree/master/recognition/tasks/dctdp): ResNet50 pretrained using the VGGFACE2 dataset; download from [MAP2V](https://github.com/Beauty9882/MAP2V) and place it in `checkpoints/`.

* [PartialFace](https://github.com/Tencent/TFace/tree/master/recognition/tasks/partialface): IResNet50 pretrained using the MS1MV2 dataset; download from [MAP2V](https://github.com/Beauty9882/MAP2V) and place it in `checkpoints/`.

*We are not the authors of any of these references.*

## Usage

### Step (a) - Robust Latent Code Initialization
```
python initialize.py --output latents/demo --V 10 --tau_K 0.999 --tau_D 0.999
```
We have released 1,000 robust latent codes, available in `latents/best` and `latents/best.mat`.

### Step (b) and (c) - Model Inversion Attack
```
python DiffMI.py --input_target imgs/target/celeba --output imgs/inversion/partialface_celeba --model partialface --latent latents/best --N 3 --attack white --eps 35 --tau_C 0.98 --tau_F 0.28
```

### Cross-Model Evaluation
```
python evaluate.py --model arcface --tau_F 0.23 --input_eval imgs/inversion/partialface_celeba --input_target imgs/target/celeba
```

## User Study

We provide the full user study questionnaire in `UserStudy.pdf`.