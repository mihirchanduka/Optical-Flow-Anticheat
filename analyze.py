import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import cv2
import sys
import re
import colorama
from colorama import Fore, Style

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()     # inheriting from existing RNN class
        self.num_layers = num_layers    # number of input layers
        self.hidden_size = hidden_size  # number of hidden players

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # creating LSTM layer
        self.fc = nn.Linear(hidden_size, num_classes)                               # creating linear output layer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # x -> (batch_size, seq_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out -> (batch_size, seq_size, input_size) = (N, 50, 512)
        out = out[:, -1, :]
        # out -> (N, 512)
        out = self.fc(out)


        return torch.sigmoid(out) # returning one forward step of the NN

''' #Old
# Function to calculate optical flow between two frames
def calc_optical_flow(prev_frame, cur_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Updated Values (Probably Less Accurate)
    #flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 5, 20, 10, 5, 1.5, 0)
    # Calculate magnitude of optical flow vectors
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    return mag
'''

#Visually dipslays optical flow vevctors 
def calc_optical_flow(prev_frame, cur_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Alternative Values (Probably Less Accurate)
    #flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 5, 20, 10, 5, 1.5, 0)
    #flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.3, 5, 25, 5, 5, 1.2, 0)
    
    # Calculate magnitude and angle of optical flow vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize magnitude to be between 0 and 255
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert angle to hue and magnitude to value in HSV color space
    hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV image to BGR for display
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Display optical flow magnitude image
    cv2.imshow('Optical Flow Magnitude', bgr)
    cv2.waitKey(1)

    return mag



if(len(sys.argv) != 3):
    print("Usage: python analyze.py <clips_path> <frames_path>")
    sys.exit()

clips_path = sys.argv[1]
frames_path = sys.argv[2]


if(not os.path.exists(clips_path)):
    print("Clips file is invalid.")
    sys.exit()

if(not os.path.exists(frames_path)):
    print("Frames directory is invalid.")
    sys.exit()



# Device configuration
device = torch.device('cpu')

#Load Custom Trained model pretrained on ImageNet and remove the last fully connected layer
model = torch.load('./models/model_trained.pt')
model.eval()
model.fc = nn.Identity()    
model.to(device)

# Load clips file
all_clips = torch.load(clips_path)

# Define threshold for suspicious clips
threshold = 10

# Analyze each clip in the clips file
results = []
for clip_idx, clip in enumerate(all_clips):
    print(f"Analyzing clip {clip_idx}...")

    # Load frames for the clip
    clip_folder = os.path.join(frames_path, f"{clip_idx}")
    frames = [cv2.imread(os.path.join(clip_folder, f)) for f in sorted(os.listdir(clip_folder))]
    # Determine Region of Interest (ROI) where the player's crosshair is usually located
    roi = (0, 0, 224, 224) # (x1, y1, x2, y2) coordinates of ROI

    # Initialize list to store magnitudes of optical flow vectors for each frame
    magnitudes = []
    opflo_cheater = 0 
    opflo_legit = 0 
    rnn_cheater = 0 
    rnn_legit = 0
    # Calculate optical flow between consecutive frames and analyze the data
    for i in range(1, len(frames)):
        frame_count = i
        # Calculate optical flow between previous frame and current frame
        prev_frame = frames[i-1][roi[1]:roi[3], roi[0]:roi[2], :]
        cur_frame = frames[i][roi[1]:roi[3], roi[0]:roi[2], :]
        mag = calc_optical_flow(prev_frame, cur_frame)

        # Store magnitude of optical flow vectors for current frame
        magnitudes.append(mag)

        # Check if there are any abnormal movements in the player's crosshair
        if np.max(mag) > threshold:
            # Calculate certainty value based on magnitude of optical flow vectors
            certainty = np.mean(mag) / np.max(mag)

            # Process Clip Through RNN as verification
            with torch.no_grad():
                for i in range(len(all_clips)):
                    curr_test = all_clips[i].unsqueeze(0)
                    output = model(curr_test)
                    
            # Convert output to a probability value between 0 and 1
            prob = torch.sigmoid(output)[0][0].item() 

            if certainty > 0.55:
                print(f"Clip: {clip_idx} Frame: {frame_count}| {Fore.RED}Optical Flow thinks the player is cheating on this frame with {certainty:.2f} certainty{Style.RESET_ALL}")
                opflo_cheater +=1   
            else: 
                print(f"Clip: {clip_idx} Frame: {frame_count}| {Fore.GREEN}Optical flow thinks player is not cheating on this frame with with {1 -certainty:.2f} certainty{Style.RESET_ALL}")
                opflo_legit +=1 
            if prob > 0.60:
                print(f"Clip: {clip_idx} Frame: {frame_count}| {Fore.RED}RNN thinks the player is cheating on this frame with with {prob:.2f} certainty{Style.RESET_ALL}")
                rnn_cheater +=1 
            else: 
                print(f"Clip: {clip_idx} Frame: {frame_count}| {Fore.GREEN}RNN thinks player is not cheating on this frame with with {1 - prob:.2f} certainty{Style.RESET_ALL}")
                rnn_legit +=1  
opflo_avg = opflo_cheater/(opflo_cheater+opflo_legit) * 100
rnn_avg = rnn_cheater/(rnn_cheater+rnn_legit) * 100
print("Optical flow is " + str(opflo_avg) + "% sure the player is cheating")
print("RNN is " + str(rnn_avg) + "% sure the player is cheating")