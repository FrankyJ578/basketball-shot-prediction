import cv2 as cv
import numpy as np
import os
import h5py

from collections import deque
from PIL import Image

# Global variables
VIDEO_LOC = 'videos'
SAVE_VIDEO_SEG_LOC = 'saved_video_inputs'
SHOT_FRAMES_LOC = 'shot_frames'
SHOOTER_CENTERS_LOC = 'shooter_center'

HEIGHT = 1080
WIDTH = 720

def list_files_from_dir(dir):
    """ List all the files inside of a given directory"""

    return [os.path.join(dir, f) for f in os.listdir(dir)]

def save_video_frames(video_filepath, shot_frames_filepath,
                      shooter_center_filepath, save_filename):
    """
    Takes a video file (eg. mp4) and parses out frames requested
    and gets the numpy array representation.
    """

    cap = cv.VideoCapture(video_filepath)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print('frame_count:{}\nframe_width: {}\nframe_height: {}'.format(frame_count, frame_width, frame_height))

    needed_frames = deque()

    # Populate needed_frames
    get_shot_frames(shot_frames_filepath, needed_frames)

    # Case if there are no frames needed from this video, or
    # there were no frames provided in .txt
    if not needed_frames:
        cap.release()
        print('No Frames provided in {}'.format(shot_frames_filepath))
        return

    # Get center for downsized frame
    center_x, center_y = get_shooter_center(shooter_center_filepath)

    # Case if there is no center provided for this video
    if center_x is None or center_y is None:
        cap.release()
        print('No center provided in {}'.format(shooter_center_filepath))
        return

    # Configure the bounds of the crop bounding box
    left_x, right_x, top_y, bot_y = configure_crop_bounds(center_x, center_y)

    cur_start, cur_end = needed_frames.popleft()
    cur_frame = 0
    data = []
    one_shot_data = dict()
    while cap.isOpened():
        ret, frame = cap.read()

        # Move past random (ret, frame) -> (False, None) that aren't
        # the end of the video.
        if not ret and cur_frame >= frame_count:
            break
        elif not ret:
            continue

        if cur_frame >= cur_start and cur_frame < cur_end:
            add_frame(cur_frame, cur_start, one_shot_data, frame,
                      left_x, right_x, top_y, bot_y)
        elif cur_frame == cur_end and needed_frames:
            print(cur_frame)
            add_frame(cur_frame, cur_start, one_shot_data, frame,
                      left_x, right_x, top_y, bot_y)
            cur_start, cur_end = needed_frames.popleft()
            append_to_data(one_shot_data, data)

            one_shot_data = dict()
        elif cur_frame >= cur_end and not needed_frames:
            add_frame(cur_frame, cur_start, one_shot_data, frame,
                      left_x, right_x, top_y, bot_y)
            append_to_data(one_shot_data, data)
            print('Finished Processing all necessary frames.')
            break

        cur_frame += 1

    cap.release()
    save_filepath = os.path.join(SAVE_VIDEO_SEG_LOC, save_filename)

    with h5py.File(save_filepath, 'w') as f:
        dset = f.create_dataset(save_filename, data=np.array(data))

def configure_crop_bounds(center_x, center_y):
    """ Return the bounds of the new frame that will be cropped
    from the original frame."""

    left_x, top_y = center_x - int(WIDTH/2), center_y - int(HEIGHT/2)
    right_x, bot_y = left_x + WIDTH, top_y + HEIGHT
    return left_x, right_x, top_y, bot_y

def get_shooter_center(shooter_center_filepath):
    """ Read from shooter_centers_filepath to get the shooter's
    center, which will be used to crop the frames into much
    smaller and more manageable frames."""

    center_x, center_y = None, None

    with open(shooter_center_filepath, 'r') as center_fp:
        line = center_fp.readline()
        center_x, center_y = line.strip().split(' ')
        center_x, center_y = int(center_x), int(center_y)

    return center_x, center_y

def get_shot_frames(shot_frames_filepath, needed_frames):
    """ Read from shot_frames_filepath to get start
    and end frames for shots within the video."""

    with open(shot_frames_filepath, 'r') as shots_fp:
        line = shots_fp.readline()
        while line:
            first_frame, last_frame = line.strip().split(' ')
            needed_frames.append((int(first_frame), int(last_frame)))
            line = shots_fp.readline()

def add_frame(cur_frame, cur_start, one_shot_data, frame,
              left_x, right_x, top_y, bot_y):
    """ Adds one frame to appropriate bin"""

    frame = modify_frame(frame, left_x, right_x, top_y, bot_y)
    bin = (cur_frame - cur_start) % 4
    one_shot_data.setdefault(bin, [])
    one_shot_data[bin].append(frame)

def append_to_data(one_shot_data, data):
    """ Takes the frames we gathered from one shot and adds them
    to overall data store for the video.
    """
    for key in one_shot_data:
        data.append(one_shot_data[key])

def modify_frame(frame, left_x, right_x, top_y, bot_y):
    """ Grayscales and crops the frame to be a more manageable size"""

    new_frame = np.dot(frame[...,:3], [.2989, .5870, .1140])
    new_frame = new_frame[top_y:bot_y, left_x:right_x]
    return new_frame

def process_all_videos():
    video_files = sorted(list_files_from_dir(VIDEO_LOC))
    shot_frames_files = sorted(list_files_from_dir(SHOT_FRAMES_LOC))
    shooter_center_files = sorted(list_files_from_dir(SHOOTER_CENTERS_LOC))
    meta_file = set(map(lambda x: x.split('/')[-1].replace('.npy', ''),
                    list_files_from_dir(SAVE_VIDEO_SEG_LOC)))

    for i in range(len(video_files)):
        filename = video_files[i].split('/')[-1].replace('.MP4', '')

        # TODO: Change this when needing to process all files
        if filename == '.DS_Store' or filename in meta_file or filename != 'sample':
            continue

        print(repr(i) + ":" + repr(filename))

        # i-1 for shot_frames_files because video_files first elem is .DS_Store
        save_video_frames(video_files[i], shot_frames_files[i-1],
                          shooter_center_files[i-1], filename)

process_all_videos()
