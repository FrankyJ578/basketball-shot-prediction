import cv2 as cv
import numpy as np
import os
from collections import deque

# Global variables
VIDEO_LOC = 'videos'
TEST_VIDEO = 'small_test.MP4'
SAVE_VIDEO_SEG_LOC = 'saved_video_inputs'
SHOT_FRAMES_LOC = 'shot_frames'

def list_files_from_dir(dir):
    """ List all the files inside of a given directory"""

    return [os.path.join(dir, f) for f in os.listdir(dir)]

def save_video_frames(video_filepath, shot_frames_filepath, save_filename):
    """
    Takes a video file (eg. mp4) and parses out frames requested
    and gets the numpy array representation.
    """

    cap = cv.VideoCapture(video_filepath)
    total_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(total_length)
    needed_frames = deque()

    # Read from shot_frames_filepath
    with open(shot_frames_filepath, 'r') as shots_fp:
        line = shots_fp.readline()
        while line:
            first_frame, last_frame = line.strip().split(' ')
            needed_frames.append((int(first_frame), int(last_frame)))
            line = shots_fp.readline()

    # Case if there are no frames needed from this video, or
    # there were no frames provided in .txt
    if not needed_frames:
        cap.release()
        print('No Frames provided in {}'.format(shot_frames_filepath))
        return

    cur_start, cur_end = needed_frames.popleft()
    cur_frame = 0
    data = []
    one_shot_data = dict()
    while cap.isOpened():
        ret, frame = cap.read()

        # Move past random (ret, frame) -> (False, None) that aren't
        # the end of the video.
        if not ret and cur_frame >= total_length:
            break
        elif not ret:
            continue

        if cur_frame >= cur_start and cur_frame < cur_end:
            add_frame(cur_frame, cur_start, one_shot_data, frame)
        elif cur_frame == cur_end and needed_frames:
            print(cur_frame)
            add_frame(cur_frame, cur_start, one_shot_data, frame)
            cur_start, cur_end = needed_frames.popleft()
            append_to_data(one_shot_data, data)

            one_shot_data = dict()
        elif cur_frame >= cur_end and not needed_frames:
            add_frame(cur_frame, cur_start, one_shot_data, frame)
            append_to_data(one_shot_data, data)
            print('Finished Processing all necessary frames.')
            break

        cur_frame += 1

    print(cur_frame)
    cap.release()
    save_filepath = os.path.join(SAVE_VIDEO_SEG_LOC, save_filename)
    np.save(save_filepath, np.array(data))

def add_frame(cur_frame, cur_start, one_shot_data, frame):
    bin = (cur_frame - cur_start) % 4
    one_shot_data.setdefault(bin, [])
    one_shot_data[bin].append(frame)

def append_to_data(one_shot_data, data):
    for key in one_shot_data:
        data.append(one_shot_data[key])

def process_all_videos():
    video_files = sorted(list_files_from_dir(VIDEO_LOC))
    shot_frames_files = sorted(list_files_from_dir(SHOT_FRAMES_LOC))
    meta_file = set(map(lambda x: x.split('/')[-1].replace('.npy', ''),
                    list_files_from_dir(SAVE_VIDEO_SEG_LOC)))

    for i in range(len(video_files)):
        filename = video_files[i].split('/')[-1].replace('.MP4', '')

        # TODO: Change this when needing to process all files
        if filename == '.DS_Store' or filename in meta_file or filename != 'sample':
            continue

        print(repr(i) + ":" + repr(filename))

        # i-1 for shot_frames_files because video_files first elem is .DS_Store
        save_video_frames(video_files[i], shot_frames_files[i-1], filename)

process_all_videos()
