from typing import Tuple
from torch import Tensor
import math
import numpy as np
import torch
from utils import cos_similarity_score


def cal_eer(
        features: Tensor,
        labels: Tensor,
        fpr_max: float = None,
        ) -> Tuple[float, float, float]:
    from sklearn.metrics import roc_curve,roc_auc_score
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    geniue_index1 = list()
    geniue_index2 = list()
    imposter_index1 = list()
    imposter_index2 = list()
    for i in range(labels.shape[0]):
        for j in range(i+1, labels.shape[0]):
            if labels[i]==labels[j]:
                geniue_index1.extend([i])
                geniue_index2.extend([j])
            else:
                imposter_index1.extend([i])
                imposter_index2.extend([j])
    if len(geniue_index1) ==0 or len(imposter_index1)==0:
        raise RuntimeError("single class or single sample dataset")
    geniue_score = cos_similarity_score(features[geniue_index1], features[geniue_index2]).cpu()
    imposter_score = cos_similarity_score(features[imposter_index1], features[imposter_index2]).cpu()
    auc =  roc_auc_score([1]*len(geniue_score)+[0]*len(imposter_score), torch.cat((geniue_score,imposter_score),0))
    fprs, tprs, thresholds = roc_curve([1]*len(geniue_score)+[0]*len(imposter_score), torch.cat((geniue_score,imposter_score),0), pos_label=1)
    if fpr_max is None:
        # EER
        eer = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
        thres = float(interp1d(fprs, thresholds)(eer))
        error = eer
    else:
        # FAR<?
        fpr = fpr[np.where(fpr < fpr_max)[0][-1]]
        thres = float(interp1d(fprs, thresholds)(fpr))
        fnr = 1 - float(interp1d(fprs, tprs)(fpr))
        error = fnr
    
    return auc, error, thres

def false_rate(
        featuresA: Tensor,
        labelsA: Tensor,
        featuresB: Tensor,
        lablesB: Tensor,
        thres: float,
        ) -> Tuple[float, float]:
    geniue_indexA = list()
    geniue_indexB = list()
    imposter_indexA = list()
    imposter_indexB = list()
    for i in range(labelsA.shape[0]):
        for j in range(lablesB.shape[0]):
            if labelsA[i]==lablesB[j]:
                geniue_indexA.extend([i])
                geniue_indexB.extend([j])
            else:
                imposter_indexA.extend([i])
                imposter_indexB.extend([j])
    geniue_score = cos_similarity_score(featuresA[geniue_indexA], featuresB[geniue_indexB])
    imposter_score = cos_similarity_score(featuresA[imposter_indexA], featuresB[imposter_indexB])
    frr = -1
    if len(geniue_score) > 0:
        frr = (geniue_score < thres).float().mean().item()
    far = -1
    if len(imposter_score) > 0:
        far = (imposter_score >= thres).float().mean().item()
        
    return far, frr

def asr(
        test: Tensor,
        target: Tensor,
        thres: float,
        interval: int = 10,
        ) -> Tuple[float, float, float]:
    type2_test = list()
    type2_target = list()
    for i in range(len(test)):
        type2_test.extend([i]*(interval-1))
        tmp = list(range(math.floor(i/interval)*interval, math.floor(i/interval)*interval+interval))
        tmp.pop(i%interval)
        type2_target.extend(tmp)
    type1_score = cos_similarity_score(test, target)
    print(f"Similarity = {type1_score.mean().item()}")
    type1_acc = (type1_score >= thres).float().mean().item()
    type2_score = cos_similarity_score(test[type2_test], target[type2_target])
    type2_acc = (type2_score >= thres).float().mean().item()
    acc = (type1_acc + type2_acc * (interval-1)) / interval

    return acc, type1_acc, type2_acc