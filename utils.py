from typing import Tuple,Any
from torch import Tensor
import os
import math
import numpy as np
import torch


def load_samples(
        base_path: str,
        start: int = 0,
        end: int = None,
        shape: Tuple[int, int] = None,
        ) -> Tuple[Tensor, Tensor]:
    from PIL import Image
    files = os.listdir(base_path)
    if start is None:
        start = 0
    if end is None:
        end = len(files)
    images, labels = [], []
    for i in range(start, end):
        # get filename and label
        file = [n for n in files if f"{i:05d}_" in n][0]
        label = int(file.split(".")[0].split("_")[-1])
        # open file
        path = os.path.join(base_path, file)
        image = Image.open(path)
        if shape is not None:
            image = image.resize(shape)
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        assert image.ndim == 3
        image = np.transpose(image, (2, 0, 1))
        images.append(image)
        labels.append(label)
    images_ = np.stack(images)
    labels_ = np.array(labels)
    images_ = images_ / 255
    
    return Tensor(images_), Tensor(labels_).long()

def save_all_images(
        imgs: Tensor,
        labels: Any,
        output_path: str,
        start_idx: int = 0,
        ) -> None:
    from torchvision.utils import save_image
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(len(imgs)):
        save_image(imgs[i], f'{output_path}/%05d_%d.png'%(i+start_idx,labels[i]))

def imgs_resize(
        imgs: Tensor,
        shape: Tuple[int, int],
        ) -> Tensor:
    from torchvision.transforms import Resize
    transform = Resize(shape, antialias=True)
    images = transform(imgs)
    
    return images

def load_model(model_name: str):
    if model_name == 'arcface':
        from models.insightface.iresnet import iresnet100
        model = iresnet100(pretrained=True)
        shape = (112,112)
    elif model_name == 'facenet':
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained='vggface2')
        shape = (160,160)
    elif model_name == 'partialface':
        from models.PartialFace.PartialFace import PartialFaceModel
        model = PartialFaceModel()
        pth_files = ["checkpoints/partialface_Backbone_Epoch_40_checkpoint.pth", "checkpoints/partialface_HEAD_Epoch_40_Split_0_checkpoint.pth", "checkpoints/partialface_META_Epoch_40_checkpoint.pth"]
        for pth_file in pth_files:
            state_dict = torch.load(pth_file, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        shape = (112,112)
    elif model_name == 'DCTDP':
        from models.DCTDP.DCTDP import DCTDP_model
        _, _, model = DCTDP_model()
        shape = (112,112)
    else:
        raise ValueError("unsupported model")
    model.eval()
    
    return model, shape

def predict(
        model,
        imgs: Tensor,
        batch_size: int = None,
        ) -> Tensor:
    device = imgs.device
    count = len(imgs)
    if batch_size is None:
        batch_size = count
    batches = math.ceil(count/batch_size)
    logits = Tensor([]).to(device)
    for b in range(batches):
        if b == batches-1:
            idx = range(b*batch_size, count)
        else:
            idx = range(b*batch_size, (b+1)*batch_size)
        with torch.no_grad():
            logits_batch = model.to(device)(imgs[idx])
        logits = torch.cat((logits, logits_batch), 0)
    
    return logits

def cos_similarity_score(
        featuresA: Tensor,
        featuresB: Tensor,
        ) -> Tensor:
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(featuresA, featuresB)
    
    return similarity