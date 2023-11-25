from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randint

from flask import Flask, request, render_template, send_from_directory, jsonify


# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from scipy import misc
from skimage.transform import resize
import imageio
import cv2
import numpy as np
import facenet
import detect_face
import os
import pickle
import json
import time
import datetime
import math
from scipy.ndimage import rotate
from sklearn.svm import SVC
# from attendence import *
from write_file import *
from datapreprocess import *
from extractframes import *
from frames import *
from data_processing import *


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload_new.html")


#Upload new Faces
@app.route("/upload", methods=["POST"])
def upload():

    label = request.form['name']
    print("Name: ", label)

    id = request.form['id']
    #
    a= store_id(label,id)
    a.write_json()
    vid_dir = os.getcwd() + '/datasets/videos/' + str(id)
    video_dir = os.path.expanduser(vid_dir)
    
    # raw_dir = os.getcwd() + '/datasets/raw/' + str(id)
    if not os.path.exists(video_dir) and a.make_dir:
        os.mkdir(video_dir) 

    else:
        print("Couldn't create upload directory: {}".format(video_dir))


    for upload in request.files.getlist("file"):
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([video_dir, filename])
        upload.save(destination)
    print("Destination: ", destination)

    return jsonify(status="Video Uploaded")


#Preprocess new Faces
@app.route("/dataprocess", methods=["POST", "GET"])
def dataprocess():
    video_dir = os.getcwd() + '/datasets/videos/'
    file_name = 'frames_record.txt'

    frames(video_dir, file_name)

    #Align Data
    output_dir_path = os.getcwd() + '/datasets/aligned'
    output_dir = os.path.expanduser(output_dir_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datadir = os.getcwd() + '/datasets/raw'
    dataset = facenet.get_dataset(datadir)

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.getcwd() + '/align')

    minsize = 75  # minimum size of face
    threshold = [0.6, 0.8, 0.92]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    image_size = 182

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    print('Goodluck')


    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = imageio.imread(image_path)
                    print('read data dimension: ', img.ndim)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                        print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
                    print('after data dimension: ', img.ndim)

                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('detected_face: %d' % nrof_faces)
                    if nrof_faces == 1:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        # if nrof_faces > 1:
                        #     bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        #     img_center = img_size / 2
                        #     offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                        #                          (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                        #     offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        #     index = np.argmax(
                        #         bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                        #     det = det[index, :]
                        det = np.squeeze(det)
                        bb_temp = np.zeros(4, dtype=np.int32)

                        bb_temp[0] = det[0]
                        bb_temp[1] = det[1]
                        bb_temp[2] = det[2]
                        bb_temp[3] = det[3]

                        try:
                            cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                            # print("Cropped Image: ", cropped_temp)
                            # scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')
                            scaled_temp = resize(cropped_temp, (image_size, image_size), mode='reflect', anti_aliasing=True)
                            scaled_temp = (scaled_temp * 255).astype(np.uint8)
                        except:
                            pass

                        nrof_successfully_aligned += 1
                        imageio.imwrite(output_filename, scaled_temp)
                    else:
                        print('Unable to align "%s"' % image_path)

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    aligned_dir = os.getcwd() + '/datasets/aligned/'
    processed_file_record = 'processing_record.txt'
    data_processing(aligned_dir, processed_file_record)

    return jsonify(status=True, message='Ready for training')


#Train Data
@app.route("/train", methods=["POST"])
def train():

    with tf.Graph().as_default():

        with tf.Session() as sess:

            datadir = os.getcwd() + '/datasets/aligned'
            dataset = facenet.get_dataset(datadir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            print('Loading feature extraction model')
            modeldir = os.getcwd() + '/models/facenet/20170512-110547.pb'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            batch_size = 20
            image_size = 160
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename = os.getcwd() + '/models/classifier/classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)

            # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True, tol = 0.5)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]
            print("iam classname",class_names)
            # Saving classifier model

            open("train.txt", "w").writelines([l for l in open("data.txt").readlines()])

            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
                # joblib.dump((model, class_names), self.classifier_filename_exp)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            print('Goodluck')

    return jsonify(status="Trained")

class LoadModel():
    """  Importing and running isolated TF graph """
    def __init__(self):
        # Create local graph and use it in the session
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))

            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, os.getcwd() + '/align')

                self.modeldir = os.getcwd() + '/models/facenet/20170512-110547.pb'
                facenet.load_model(self.modeldir)

                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                self.classifier_filename = os.getcwd() + '/models/classifier/classifier.pkl'
                self.classifier_filename_exp = os.path.expanduser(self.classifier_filename)

                with open(self.classifier_filename_exp, 'rb') as infile:
                    (self.model, self.class_names) = pickle.load(infile)
                    print('load classifier file-> %s' % self.classifier_filename_exp)


    def embedding_tensor(self):
        return(self.embedding_size)

    def nets(self):
        return(self.pnet, self.rnet, self.onet)


    def predict(self, data, emb_array):
        """ Running the activation operation previously imported """

        feed_dict = {self.images_placeholder: data, self.phase_train_placeholder: False}
        emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
        predictions = self.model.predict_proba(emb_array)
        print(emb_array)

        return(predictions)


model = LoadModel()
minsize = 75  # minimum size of face
threshold = [0.6, 0.8, 0.92]  # three steps's threshold
factor = 0.709  # scale factor
image_size = 182
input_image_size = 160
pnet, rnet, onet = model.nets()
embedding_size = model.embedding_tensor()


with open('train.txt') as json_file:
    HumanNames = json.load(json_file)

l = list(HumanNames)
l.sort()

# Crop Padding
left = 1
right = 1
top = 1
bottom = 1


#Recognize
@app.route("/recognize", methods=["GET","POST"])
def recognize():
    for upload in request.files.getlist("file"):

        img_array = np.array(bytearray(upload.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)

        nameList = []
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)

        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Face Detected: %d' % nrof_faces)

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]

            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

            for i in range(nrof_faces):
                emb_array = np.zeros((1, embedding_size))

                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                # inner exception
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    continue

                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                cropped[i] = facenet.flip(cropped[i], False)
                # print(cropped)
                # scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                temp_img = resize(cropped[i], (image_size, image_size), mode='reflect', anti_aliasing=True)
                scaled.append((temp_img * 255).astype(np.uint8))
                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                       interpolation=cv2.INTER_CUBIC)
                scaled[i] = facenet.prewhiten(scaled[i])
                scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))

                #Call function inside the loaded model
                predictions = model.predict(scaled_reshape[i], emb_array)

                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                # print(best_class_indices)

                try:
                    best_class = l[best_class_indices[0]]
                except:
                    print(best_class_indices[0])
                result_names = HumanNames[best_class]

                print("ID: ", best_class, result_names)
                print(best_class_probabilities)

                if best_class_probabilities >= 0.75:

                    name = result_names
                    id = int(best_class)
                    now = datetime.datetime.now()
                    checkin_time = now.strftime("%H:%M:%S")

                    mytime = checkin_time.split(':',-1)
                    mytime = int(mytime[0])
                    if mytime < 15:
                        name = 'Hello ' + name
                    else:
                        name = 'Bye ' + name
                    classpath = 'datasets/recognized_data/{}'.format(str(id))
                    checkin_date = datetime.date.today().strftime("%B:%d:%Y")
                    if not os.path.exists(classpath):
                        os.mkdir(classpath)
                    cv2.imwrite(('{}/{}_{}_{}.jpg'.format(classpath,result_names, str(checkin_time), checkin_date)), frame)
                    accuracy = str(best_class_probabilities)
                    response = {"id":id,"name":name, "time":checkin_time, "accuracy":accuracy}
                    nameList.append(response)

                else:
                    # name = "Unknown"
                    # id = None
                    # checkin_time = None
                    # accuracy = None
                    now = datetime.datetime.now()
                    checkin_time = now.strftime("%H:%M:%S")
                    checkin_date = datetime.date.today().strftime("%B:%d:%Y")
                    classpath = 'datasets/non_recognized/'
                    cv2.imwrite(('{}/{}_{}.jpg'.format(classpath, str(checkin_time), checkin_date)), frame)


        else:
            return None


    return jsonify(response=nameList)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, debug=True)
