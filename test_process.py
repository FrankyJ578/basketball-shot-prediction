import numpy as np
from PIL import Image
import h5py

TEST_FILE = 'saved_video_inputs/sample'

if __name__=='__main__':
    with h5py.File(TEST_FILE, 'r') as f:
        data = f['sample'][:]

    print(data.shape)

    for i in range(8):
        Image.fromarray(data[0][i]).show()
