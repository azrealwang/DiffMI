import argparse
import torch
from utils import load_samples,load_model,predict
from metrics import cal_eer,false_rate,asr

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='facenet, arcface, partialface, DCTDP', type=str, required=True)
    parser.add_argument('--tau_F', help='threshold of face recognition', type=float, default=None)
    parser.add_argument('--input_eval', help='evaluated image path', type=str, required=True)
    parser.add_argument('--input_target', help='target image path', type=str, required=True)
    parser.add_argument('--interval', help='how many faces per identity', type=int, default=5)
    parser.add_argument('--start_idx', help='start index', type=int, default=0)
    parser.add_argument('--end_idx', help='end index', type=int, default=10)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=None)
    
    args = parser.parse_args()

    return args

def evaluate(args):
    ## Load Model
    model, shape = load_model(args.model)

    ## Load inputs
    x_test, _ = load_samples(args.input_eval, args.start_idx, args.end_idx, shape)
    test = predict(model, x_test.to(args.device), args.batch_size,)
    x_target, y_target = load_samples(args.input_target, args.start_idx, args.end_idx, shape)
    target = predict(model, x_target.to(args.device), args.batch_size)
    
    ## Model accuracy
    if args.tau_F is None:
        _, eer, args.tau_F = cal_eer(target, y_target) # target accuracy
        print(f"EER = {eer}, threshold = {args.tau_F}")
    else:
        far, frr = false_rate(target, y_target, target, y_target, args.tau_F)  # target accuracy
        print(f"FAR = {far}, FRR = {frr}")
    ## Evaluation accuracy
    acc, type1, type2 = asr(test, target, args.tau_F, args.interval)
    print(f"ASR = {acc}, Type I Accuracy = {type1}, Type II Accuracy = {type2}")

    return args.tau_F

if __name__ == "__main__":
    args = parse_args_and_config()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args)
    evaluate(args)