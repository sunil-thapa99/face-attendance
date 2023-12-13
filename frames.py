import cv2
import os
import glob
import json
from extractframes import *

file_name = 'frames_record.txt'
video_dir = os.getcwd() + '/datasets/videos'


def frames(video_dir, file_name):
    try:
        with open(file_name) as json_file:
            content = json.load(json_file)
    except:
        with open(file_name, 'w')as json_file:
            json.dump({}, json_file)
            content = json.load(json_file)
        raise

    file_list = os.listdir(video_dir)
    file_list = [file for file in file_list if file != '.DS_Store']
    for root in file_list:
        vid_file = root.split('/')[-1]
        for vid in os.listdir(os.path.join(video_dir, root)):
            video = os.path.join(video_dir, os.path.join(root, vid))
            if str(vid_file) not in content:
                extract_frames(video, vid_file)


if __name__ == "__main__":
    frames(video_dir, file_name)

    # for root, dirs, files in os.walk(video_dir):
    #     print("Root: ", root)

    #     cls = root.split('/')[-1]
    #     print("Class: ", cls)

    #     for vid in os.listdir(root):
    #         video = os.path.join(root, vid)
    #         print("Video: ", video)
    #         if str(cls) not in content:
    #             extract_frames(video, cls)
