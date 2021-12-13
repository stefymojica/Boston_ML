import os
import torch

import pretrainedmodels

import numpy as np
import pandas as pd

import torch.nn as nn

class SEResNext50_32X4D(nn.Module):
    def __init__(self, pretrained="imsgenet"):
        super(SEResNext50_32X4D, self).__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4"]
