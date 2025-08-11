import os
import argparse
import torch
from scipy.io import savemat
from scipy.stats import normaltest
from attacks.diffusion import reconstruct
from utils import save_all_images

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='a folder including initial faces, and a .mat including initial latent codes', type=str, required=True)
    parser.add_argument('--V', help='number of candidates', type=int, default=1000)
    parser.add_argument('--tau_K', help='threshold of K-Square test', type=float, default=0.999)
    parser.add_argument('--tau_D', help='threshold of MTCNN detection', type=float, default=0.999)

    args = parser.parse_args()

    return args

def normalize(xs):
    # normalize to N(0,1)
    mean = torch.mean(xs,dim=[-1,-2])
    std = torch.std(xs,dim=[-1,-2])
    count = len(xs)
    for i in range(count):
        for j in range(3):
            xs[i,j] = (xs[i,j]-mean[i,j])/std[i,j]

    return xs

def mtcnn(
        input_path: str, 
        image_size: int = 160,
        start_idx: int = 0,
        end_idx: int = None,
        device: str = 'cpu',
        ):
    from facenet_pytorch import MTCNN
    from PIL import Image
    mtcnn = MTCNN(image_size=image_size, device=device, post_process=False)
    files = os.listdir(input_path)
    probs = []
    if end_idx is None:
        end_idx = len(files)
    for i in range(start_idx, end_idx):
        file = [n for n in files if f"{i:05d}_" in n][0]
        input_file = os.path.join(input_path, file)
        image = Image.open(input_file)
        _, prob = mtcnn(image, return_prob=True)
        if prob is None:
            prob = 0
        probs.append(prob)

    return probs


def initial(args):
    g = torch.Tensor([])
    x = torch.Tensor([])
    i = 0
    while (i < args.V):
        print(f"Processing {i+1}...")
        j = 0
        g_i = torch.Tensor([])
        while (j < 3):
            g_ij = torch.randn((1,256,256))
            p_K = normaltest(g_ij.numpy().flatten()).pvalue # K-Square test
            if p_K >= args.tau_K:
                g_i = torch.cat((g_i, g_ij), 0)
                j += 1
            else:
                continue
        g_i = g_i.unsqueeze(0)
        g_i = normalize(g_i)
        p_K = normaltest(g_i.numpy().flatten()).pvalue # K-Square test
        if p_K >= args.tau_K:
            x_i = reconstruct(g_i.to(args.device)).detach().cpu() # generation
            save_all_images(x_i, [-1], 'imgs/tmp')
            p_D = mtcnn('imgs/tmp', device=args.device) # MTCDD detection
            if p_D[0] >= args.tau_D:
                g = torch.cat((g, g_i), 0)
                x = torch.cat((x, x_i), 0)
                i += 1
            else:
               continue 
        else:
            continue
    print("Saving codes...")
    savemat(f'{args.output}.mat', mdict={'gaussian': g.numpy()})
    print("Saving images...")
    save_all_images(x, list(-1 * torch.ones(args.V)), args.output)
    
    print("Finish")
    
if __name__ == "__main__":
    args = parse_args_and_config()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args)
    initial(args)