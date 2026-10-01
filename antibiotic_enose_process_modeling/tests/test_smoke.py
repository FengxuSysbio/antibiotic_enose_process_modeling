import numpy as np
import pandas as pd
import torch
from antibiotic_enose.classification.tsfcnet import TSFCNet
from antibiotic_enose.transfer.fewshot import kennard_stone


def test_tsfc_forward_shape():
    m=TSFCNet(window=60,n_sensors=16,n_classes=3)
    y=m(torch.randn(4,60,16))
    assert tuple(y.shape)==(4,3)


def test_kennard_stone_count():
    rng=np.random.default_rng(1);F=rng.normal(size=(20,2))
    idx=kennard_stone(F,7)
    assert len(idx)==7 and len(set(idx.tolist()))==7
