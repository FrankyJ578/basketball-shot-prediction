import numpy as np
from PIL import Image

TEST_FILE = 'saved_video_inputs/sample.npy'

if __name__=='__main__':
    images = np.load(TEST_FILE)
    for i in range(8):
        Image.fromarray(images[0][i]).show()
