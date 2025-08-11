import torch
import math
from diffusers import DDIMScheduler, UNet2DModel

def reconstruct(imgs,steps:int=20,batch_size:int=1):
    device = imgs.device
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256",local_files_only=False,use_safetensors=False).to(device)
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256",local_files_only=False,use_safetensors=False)
    scheduler.set_timesteps(steps)
    assert imgs.shape[2] == model.config.sample_size
    
    x_t = imgs.clone()
    amount = len(imgs)
    batches = math.ceil(amount/batch_size)
    for b in range(batches):
        #print(f"Reconstruction: {b+1}th batch in {batches} batches")
        if b == batches-1:
            idx = range(b*batch_size,amount)
        else:
            idx = range(b*batch_size,(b+1)*batch_size)
        for t in scheduler.timesteps:
            with torch.no_grad():
                noisy_residual = model(x_t[idx],t).sample
                prev_noisy_sample = scheduler.step(noisy_residual,t,x_t[idx]).prev_sample
                x_t[idx] = prev_noisy_sample
    imgs_output = (x_t/2+0.5).clamp(0,1)

    return imgs_output