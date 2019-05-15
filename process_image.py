import cv2 as cv
import numpy as np
import os
from collections import deque

# Global variables
VIDEO_LOC = 'videos'
TEST_VIDEO = 'test.mp4'
SAVE_VIDEO_SEG_LOC = 'saved_video_inputs'

def list_files_from_dir(dir):
    """ List all the files inside of a given directory"""

    return [os.path.join(dir, f) for f in os.listdir(dir)]

def save_video_frames(video_filepath, save_filename):
    """
    Takes a video file (eg. mp4) and parses out frames requested
    and gets the numpy array representation.
    """

    cap = cv.VideoCapture(video_filepath)
    data = []

    needed_frames = deque()
    needed_frames.append((19, 20))
    needed_frames.append((25, 26))
    cur_start, cur_end = needed_frames.popleft()
    cur_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        print(ret, frame)

        if not ret:
            break

        cur_frame += 1
        if cur_frame >= cur_start and cur_frame <= cur_end:
            data.append(frame)
        elif cur_frame > cur_end and not needed_frames:
            cur_start, cur_end = needed_frames.popleft()

    cap.release()
    save_filepath = os.path.join(SAVE_VIDEO_SEG_LOC, save_filename)
    np.save(save_filepath, np.array(data))

def process_all_videos():
    video_files = list_files_from_dir(VIDEO_LOC)
    meta_file = set(map(lambda x: x.split('/')[-1].replace('.npy', ''),
                    list_files_from_dir(SAVE_VIDEO_SEG_LOC)))

    for i in range(len(video_files)):
        filename = video_files[i].split('/')[-1].replace('.mp4', '')

        if filename == '.DS_Store' or filename in meta_file:
            continue

        print(repr(i) + ":" + repr(filename))
        save_video_frames(video_files[i], filename)

process_all_videos()
