!pip install -U --upgrade tensorflow

#from tensorflow_docs.vis import embed
from tensorflow import keras
#from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

DATASET_FOLDER = '../input/deepfake-detection-challenge'
sample_train_folder = 'train_sample_videos'
TEST_FOLDER = 'test_videos'
#print the number of train and test samples of the data set 
print(f"Train sample videos: {len(os.listdir(os.path.join(DATASET_FOLDER, sample_train_folder)))}")
print(f"Test samples videos: {len(os.listdir(os.path.join(DATASET_FOLDER, TEST_FOLDER)))}")

#reads the meta data file
sample_train_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
sample_train_metadata.head()

#generates a bar plot that displays the distribution of labels in the training set
sample_train_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')
plt.show()

#returns the shape of the data frame
sample_train_metadata.shape

#selects a random sample of 3 fake videos from the training set of the dataset
fake_train_sample_video = list(sample_train_metadata.loc[sample_train_metadata.label=='FAKE'].sample(3).index)

#displays the first frame of the video as an image
def display_image_from_frame(path_to_video):
    #opens the specified video file
    img_capture = cv2.VideoCapture(path_to_video)
    #read the first frame of the video
    ret, frame = img_capture.read()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    #converts the color space of the frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame)

for video_file in fake_train_sample_video:
    display_image_from_frame(os.path.join(DATASET_FOLDER, sample_train_folder, video_file))

#selects a random sample of 3 real videos from the training set of the dataset
real_train_sample_video = list(sample_train_metadata.loc[sample_train_metadata.label=='REAL'].sample(3).index)
real_train_sample_video

#displays the first frame of the three videos as images
for video_file in real_train_sample_video:
    display_image_from_frame(os.path.join(DATASET_FOLDER, sample_train_folder, video_file))

sample_train_metadata['original'].value_counts()[0:5]

def display_image_from_video_list(video_path_list, video_folder=sample_train_folder):
    plt.figure()
    fig, ax = plt.subplots(2,3,figsize=(16,8))
    #shows the images extracted from the first 6 videos
    for i, video_file in enumerate(video_path_list[0:6]):
        #constructs the full path to the video file
        path_to_video = os.path.join(DATASET_FOLDER, video_folder,video_file)
        #capture the video
        img_capture = cv2.VideoCapture(path_to_video)
        #reads the image
        ret, frame = img_capture.read()
        #converts the color space from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #displays the image
        ax[i//3, i%3].imshow(frame)
        ax[i//3, i%3].set_title(f"Video: {video_file}")
        ax[i//3, i%3].axis('on')

same_original_fake_train_sample_video = list(sample_train_metadata.loc[sample_train_metadata.original=='atvmxvwyns.mp4'].index)
display_image_from_video_list(same_original_fake_train_sample_video)

test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATASET_FOLDER, TEST_FOLDER))), columns=['video'])

test_videos.head()

display_image_from_frame(os.path.join(DATASET_FOLDER, TEST_FOLDER, test_videos.iloc[2].video))

fake_videos = list(sample_train_metadata.loc[sample_train_metadata.label=='FAKE'].index)

from IPython.display import HTML
from base64 import b64encode

# Define a function to play a video in the notebook
def play_the_video(video_file, subset=sample_train_folder):
    # Read the contents of the video file
    video_url = open(os.path.join(DATASET_FOLDER, subset,video_file),'rb').read()
    # Encode the video content in base64 format
    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()
    # Create an HTML object that contains a video element with the specified source URL
    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)

play_the_video(fake_videos[52])