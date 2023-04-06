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