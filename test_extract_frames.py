'''This file is used to prepare training data by taking pictures frame by frame from the webcam'''

import cv2
import os
import numpy
from scipy.ndimage import rotate
import json

import ffmpeg    

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 

def extract_frames(video, id):
    rotate_img = 0
    detector = MTCNN()
    found_img = True
    i = 0

    cam = cv2.VideoCapture(video)
    rotateCode = check_rotation(video)

    while True:
        ret, im = cam.read()
        if ret:
            classpath = os.path.join('datasets/raw',str(id))
            if not os.path.exists(classpath):
                os.mkdir(classpath)

            if rotateCode is not None:
                im = correct_rotation(im, rotateCode)

            cv2.imwrite(os.path.join(classpath, str(id) + str(i) + ".jpg"), im)
        else:
            break
       

if __name__== "__main__":
    extract_frames('datasets/videos/889/2019-09-15-165946.webm', '889')
