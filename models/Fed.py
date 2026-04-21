#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import torch

def FedAvg(w):
    if len(w) == 0:
        raise ValueError("FedAvg received empty weights list")

    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        stacked = torch.stack([client_w[k] for client_w in w], dim=0)
        w_avg[k] = torch.mean(stacked, dim=0)

    return w_avg
