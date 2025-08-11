from typing import Tuple
from torch import Tensor
import argparse
import math
import time
import torch
import numpy as np
from evaluate import evaluate
from attacks.diffusion import reconstruct
from utils import load_samples,save_all_images,load_model,predict,imgs_resize,cos_similarity_score

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    # Input & Output
    parser.add_argument('--input_target', help='target image path', type=str, required=True)
    parser.add_argument('--start_idx', help='start index', type=int, default=0)
    parser.add_argument('--end_idx', help='end index', type=int, default=10)
    parser.add_argument('--output', help='output image path', type=str, required=True)
    # Target Model
    parser.add_argument('--model', help='facenet, arcface, partialface, DCTDP', type=str, required=True)
    # Step (b) - Top N Latent Code Selection
    parser.add_argument('--latent', help='candidate latent path', type=str, required=True)
    parser.add_argument('--N', help='top N selection', type=int, default=3)
    parser.add_argument('--V', help='number of candidates', type=int, default=None)
    # Step (c) - Latent Code Manipulation
    parser.add_argument('--attack', help='white or black', type=str, default='white')
    parser.add_argument('--eps', help='white-box L2 constraint', type=float, default=35)
    parser.add_argument('--sparsity', help='black-box constraint', type=float, default=0.01)
    parser.add_argument('--max_query', help='black-box constraint', type=int, default=10000)
    parser.add_argument('--seed', help='seed', type=int, default=None)
    parser.add_argument('--tau_C', help='confidence threshold', type=float, default=1)
    parser.add_argument('--disable_ranked', help='disable ranked adversary, so run all N latent codes', action='store_true')
    # Evaluation
    parser.add_argument('--tau_F', help='threshold of face recognition', type=float, default=None)
    parser.add_argument('--interval', help='how many faces per identity', type=int, default=5)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=None)
    
    args = parser.parse_args()

    return args

def best_N(
        model,
        target: Tensor,
        candidate: str,
        best_size: int = 1,
        shape: Tuple[int, int] = None,
        pool_size: int = None,
        batch_size: int = None,
        ):
    from scipy.io import loadmat
    device = target.device
    x_pool, _ = load_samples(f'{candidate}')
    x_pool = x_pool.to(device)
    if shape is not None:
        x_pool = imgs_resize(x_pool, shape)
    idx = torch.arange(len(x_pool))
    perm = torch.randperm(idx.numel())
    idx[:] = idx[perm]  
    if pool_size is None:
        pool_size = len(x_pool)
    if batch_size is None:
        batch_size = pool_size
    pool_idx = idx[:pool_size]
    pool = predict(model.to(device), x_pool[pool_idx], batch_size)
    target_batch = target.repeat(pool_size,1)
    similarity = cos_similarity_score(pool, target_batch)
    best_idx_batch = torch.sort(similarity, descending=True).indices[:best_size].cpu()
    best_idx = pool_idx[[best_idx_batch]]
    source = Tensor(loadmat(f'{candidate}.mat')['gaussian']).to(device)
    source = source[best_idx]
    x_source = x_pool[best_idx]
    
    return source, x_source, best_idx, pool_size

def DiffMI(args):
    if args.attack != 'white':
        raise ValueError("demo only supports white-box attack")
    ## Load target
    model, shape = load_model(args.model)
    x_target, y = load_samples(args.input_target, args.start_idx, args.end_idx, shape)
    target = predict(model, x_target.to(args.device), args.batch_size)

    start_time = time.time()
    for i in range(len(y)):
        print(f"Target Index: {i + args.start_idx}")
        ## Step (b) - Top N Latent Code Selection
        source, x_source, best_idx, n_query = best_N(
            model,
            target[i],
            args.latent,
            shape=shape,
            best_size=args.N,
            pool_size=args.V,
            batch_size=args.batch_size
            )
        print(f"Selected Candidates: {best_idx.tolist()}")
        ## Step (c) - Latent Code Manipulation
        pre_loss = -1
        success = False
        for j in range(args.N):
            max_query_batch = math.floor((args.max_query - n_query) / (args.N - j))
            if args.attack == 'white':
                from attacks import APGDAttack
                attack = APGDAttack(
                    model,
                    norm='L2',
                    eps=args.eps,
                    loss='fr_loss_targeted',
                    device=args.device,
                    thres=args.tau_C,
                    n_iter=100,
                    seed=args.seed,
                    shape=shape,
                    )
            elif args.attack == 'black':
                 from attacks import GreedyPixel
                 attack = GreedyPixel(
                     model,
                     eps=1,
                     max_iter=max_query_batch,
                     nb_first=int(args.sparsity * 256 * 256),
                     device=args.device,
                     thres=args.tau_C,
                     shape=shape,
                     seed=args.seed,
                     )
            else:
                raise ValueError("unsupported attack")
            x_adv, n_query_batch, loss = attack.perturb(source[j].unsqueeze(0), target[i].unsqueeze(0))
            n_query += n_query_batch
            if loss > pre_loss:
                x_source_batch = x_source[j].unsqueeze(0)
                x_adv_batch = x_adv.detach()
                pre_loss = loss
            if loss > args.tau_C:
                success = True
                if not args.disable_ranked:
                    break
            if j == args.N - 1 and success == False:
                print("This sample fails to pass margin.")
        print(f"Query Cost = {n_query}")
        x_recon_batch = reconstruct(x_adv_batch).detach()
        # Save images
        save_all_images(x_source_batch, y[i].unsqueeze(0), f'{args.output}-source', i + args.start_idx)
        save_all_images(x_recon_batch, y[i].unsqueeze(0), f'{args.output}', i + args.start_idx)
    end_time = time.time()
    print(f"Attack costs {end_time-start_time}s")
    
    ## Attack accuracy
    args.input_eval = args.output
    evaluate(args)

if __name__ == "__main__":
    args = parse_args_and_config()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args)
    DiffMI(args)