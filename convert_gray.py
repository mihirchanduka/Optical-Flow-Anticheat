import cv2
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python convert_gray.py <dir>")
    sys.exit()

dir = sys.argv[1]

if not os.path.exists(dir):
    print("Directory does not exist.")
    sys.exit()

for filename in os.listdir(dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(os.path.join(dir, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(dir, filename), gray)