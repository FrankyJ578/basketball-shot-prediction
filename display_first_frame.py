import numpy as np
from PIL import Image
import cv2 as cv

# Global variables
HEIGHT = 1080
WIDTH = 720

# CHANGE THE FILE TO BE THE VIDEO THAT YOU WANT TO GET THE
# SHOOTER'S CENTER FOR
VIDEO_FILE = 'videos/sample1.MP4'

def display():
    cap = cv.VideoCapture(VIDEO_FILE)

    ret, frame = cap.read()

    frame = np.dot(frame[...,:3], [.2989, .5870, .1140])
    Image.fromarray(frame).show()

    cap.release()

if __name__=='__main__':
    display()
