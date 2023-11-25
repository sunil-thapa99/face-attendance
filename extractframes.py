# '''This file is used to prepare training data by taking pictures frame by frame from the webcam'''

import cv2
import os
import numpy
from scipy.ndimage import rotate
import json
import subprocess

from extract_video_prop import *

file_name = 'frames_record.txt'
# content = {}

with open(file_name) as json_file:
    content = json.load(json_file)


def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = getVideoDetail(path_video_file)
    print(meta_dict)
    rotateCode = None
    # we are looking for
    try:
        rotation_val = meta_dict['TAG']['rotate']
        if int(rotation_val) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(rotation_val) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(rotation_val) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    except: 
        pass

    return rotateCode

def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 

def extract_frames(video, id):
    content.update({id:id})
    with open(file_name, 'w')as file:
        json.dump(content, file, indent=4, sort_keys=True)
    i = 0
    # print(video)

    cam = cv2.VideoCapture(video)
    rotateCode = check_rotation(video)

    while True:
        ret, im = cam.read()

        if ret:
            classpath = os.path.join('datasets/raw',str(id))
            # print(os.path.exists(classpath), classpath, id)
            if not os.path.exists(classpath):
                os.mkdir(classpath)

            if rotateCode is not None:
                im = correct_rotation(im, rotateCode)
            i += 1
            cv2.imwrite(os.path.join(classpath, str(id) + str(i) + ".jpg"), im)
        else:
            break
       

if __name__== "__main__":
    video = 'datasets/videos/378/'
    path_video = os.path.join(video, os.listdir(video)[0])
    extract_frames(path_video, '378')


# def extract_frames(video, id):
#     rotate_img = 0
#     detector = MTCNN()
#     found_img = True
#     content.update({id:id})
#     with open(file_name, 'w')as file:
#         json.dump(content, file, indent=4, sort_keys=True)
#     i = 0

#     cam = cv2.VideoCapture(video)
#     ret, im = cam.read()
#     while True:
#         cv2.imshow('IMG', im)
#         cv2.waitKey(0)
#         im = rotate(im, rotate_img)
#         face_info = detector.detect_faces(im)
#         if face_info:
#             print('BREAK')
#             break
#         else:
#             print('RUN')
#             rotate_img -= 90

#     while ret:
#         ret, im = cam.read()
#         # if found_img:
#         #     print('---------------------')
#         #     if face_info:
#         #         face_info = face_info[0]['keypoints']
#         #         r_eye = face_info['right_eye']
#         #         l_eye = face_info['left_eye']
#         #         r_mouth = face_info['mouth_right']
#         #         l_mouth = face_info['mouth_left']

#         #         if r_eye[1] > r_mouth[1] and l_eye[1] > l_mouth[1]:
#         #             rotate_img = 180
#         #         elif r_eye[1] < l_mouth[1] and l_eye[1] > r_mouth[1]:
#         #             rotate_img = 270
#         #         elif r_eye[1] > l_mouth[1] and l_eye[1] < r_mouth[1]:
#         #             rotate_img = 90
#         #         else:
#         #             rotate_img = 0
#         #         found_img = False

#         classpath = os.path.join('datasets/raw',str(id))
#         if not os.path.exists(classpath):
#             os.mkdir(classpath)

#         #Rotate image 90 degree for portrait alignment
#         try:
#             # print(rotate_img)
#             i += 1
#             im = rotate(im, rotate_img)
#         except:
#             pass
#         cv2.imwrite(os.path.join(classpath, str(id) + str(i) + ".jpg"), im)
       

# if __name__== "__main__":
#     extract_frames('datasets/videos/883/Badal Shrestha.mp4', '883')

