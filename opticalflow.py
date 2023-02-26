import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import sys
import re



def calc_optical_flow(prev_frame, cur_frame):
    # Convert all frames to grayscale

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude of optical flow vectors
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    return mag