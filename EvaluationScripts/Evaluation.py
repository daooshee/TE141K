from __future__ import print_function
import argparse
import os
from math import log10
import glob
import time
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from collections import namedtuple
import torchvision.models.vgg as vgg

plt.switch_backend('agg')



##################################################
# Parameters

# Which method to evaluate.
method_name = 'TET-GAN+(Supervised)'

# If Full==True, the evaluation will be conducted on all TE141K-E, TE141K-C, TE141K-S.
# If Full==False, the evaluation will be conducted on TE141K-S only.
Full = False

# Path to the TE141K Dataset. 
TE141K_Path = 'TE141K/'

# Path to the results of all models.
Results_Path = 'Results/'

# Use cuda or cpu.
device = torch.device("cpu")



##################################################
# VGG model and loss functions

# We use the pretrained VGG provided by pytorch.
LossOutput = namedtuple(
    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])

class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
    def forward(self, x):
        output = {}
        count = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[count] = x
                count += 1
        return output

loss_network = LossNetwork().to(device).eval()


# Gram matrix for style loss.
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


# Transfer images into tensors.
def Process(image):
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = image.float()
    image = image.mul_(2).add_(-1)
    return image.unsqueeze(0).to(device)


##################################################
# Evaluation

# If Full==True, the evaluation will be conducted on all TE141K-E, TE141K-C, TE141K-S.
# If Full==False, the evaluation will be conducted on TE141K-S only.
if Full:
    style_list = glob.glob(TE141K_Path+"*/*/val/")
else:
    style_list = glob.glob(TE141K_Path+"TE141K-S/*/val/")

avg_total_psnr = 0
avg_total_ssim = 0
avg_total_per = 0
avg_total_style = 0
total = 0

for style_path in style_list:
    # Find all content images of a given style.
    style_name = style_path.split('/')[-3]
    content_list = glob.glob(style_path+"/*.png")

    for content_path in content_list:
        # Find the corresponding result image of a given (method, style, content) pair.
        content_info = content_path.split('/')
        result = Results_Path+method_name+'/'+content_info[-3]+'_'+content_info[-1][:-4]+'.png'

        # Read the ground truth image, crop it into 256 x 256
        ground_truth = cv2.imread(content_path) / 255.
        ground_truth = ground_truth[:,320:,:]
        ground_truth = ground_truth[32:-32,32:-32,:]

        # Read the result image
        generation = cv2.imread(result) / 255.
        generation = cv2.resize(generation,(256,256))

        ##################################################
        # Quantitative Evaluation

        # PSNR.
        mse = ((ground_truth-generation)**2).mean()
        psnr = 10 * log10(1 / mse)

        # SSIM.
        ssim = compare_ssim(X=generation, Y=ground_truth, multichannel=True)

        # Perceptual and style Loss.
        err_per = 0
        err_style = 0
        Features_gt = loss_network(Process(ground_truth))
        Features_result = loss_network(Process(generation))
        for p in range(5):
            err_per += torch.mean(torch.abs(Features_gt[p]-Features_result[p]))
            Gram_gt = gram_matrix(Features_gt[p])
            Gram_gen = gram_matrix(Features_result[p])
            err_style += torch.mean(torch.abs(Gram_gt-Gram_gen))

        # Sum
        avg_total_psnr += psnr
        avg_total_ssim += ssim
        avg_total_per += err_per.item()
        avg_total_style += err_style.item()

    total += len(content_list)

print("[%s] PSNR:%f, SSIM:%f, Perceptual:%f, Style:%f" % (method_name, avg_total_psnr/total,
                                     avg_total_ssim/total, avg_total_per/total, avg_total_style/total))

