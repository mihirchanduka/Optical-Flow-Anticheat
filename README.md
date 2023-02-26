# Optical Flow Anticheat
This repository contains proof-of-concept code for detecting suspicious behavior in Counter-Strike:Global Offensive. The detection is based on a pre-trained ResNet18 model and optical flow. 
Requirements
- Python 3
- PyTorch
- OpenCV
- NumPy
- Colorama

## Required Software Packages:
- Python (3.9 +) 
- OpenCV:
Installed with: `pip install opencv-python`
- PyTorch:
Installed with: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
Or if you don't have access to an Nvidia GPU: `pip3 install torch torchvision torchaudio`
- Colorama
Insatlled with `pip install colorama`

## Usage
To analyze a set of game clips, run the following command:

```python analyze.py <clips_path> <frames_path>```

where `<clips_path>` is the path to a file containing a list of clips (in the form of numpy arrays) to be analyzed, and `<frames_path>` is the path to a directory containing the frames for each clip.

File Description
-  `analyze.py`: This is the main script for analyzing game clips and detecting suspicious behavior.
- `models/model.pt`: This is a pre-trained ResNet18 model that is used for feature extraction.


## Acknowledgements

Credits to [SrikarValluri /
aimbot-detection ](https://github.com/SrikarValluri/aimbot-detection) for auto_clip.py and save_cnn_output.py


This project is licensed under the CCO-1.0 License - see the LICENSE.md file for details.