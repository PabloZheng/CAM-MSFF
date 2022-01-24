import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


class CatGAP(nn.Module):
    def __init__(self):
        super(CatGAP, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.soft_log = nn.Softmax(dim=1)
        self.out_list = []
        self.feature_index = [2, 7, 16, 25, 34]

    def forward(self, input):
        for index in self.feature_index:
            x = self.GAP(input[index])
            x = x.view(x.size(0), -1)
            self.out_list.append(self.soft_log(x))
        return self.out_list


def accuracy(input:Tensor, targs:Tensor):
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean().cpu().detach().numpy()


def imshow_transform(image_in):
    img = np.rollaxis(image_in.squeeze().cpu().detach().numpy(), 0, 3)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    return img


class SaveFeatures():
    features=None
    def __init__(self,m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
