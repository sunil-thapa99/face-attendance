import cv2
import os
import glob
import json
from extractframes import *
from datapreprocess import *

file_name = 'processing_record.txt'
img_dir = os.getcwd() + '/datasets/aligned/'


def data_processing(img_dir, file_name):
    try:
        with open(file_name) as json_file:
            content = json.load(json_file)
    except:
        with open(file_name, 'w')as json_file:
            json.dump({}, json_file)
            content = json.load(json_file)

    for root, dirs, files in os.walk(img_dir):
        print("Root: ", root)

        cls = root.split('/')[-1]
        print("Class: ", cls)


        try:
            if str(cls) not in content:
                preprocess(root, cls, file_name)
        except:
            pass
