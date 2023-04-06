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

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# define a function to crop the center square of a given frame
def crop_square(frame):
    y, x = frame.shape[0:2] # get the height and width of the frame
    min_dim = min(y, x) # get the minimum of the two dimensions
    start_x_axis = (x // 2) - (min_dim // 2) # calculate the starting point for x-axis
    start_y_axis = (y // 2) - (min_dim // 2) # calculate the starting point for y-axis
    return frame[start_y_axis : start_y_axis + min_dim, start_x_axis : start_x_axis + min_dim] # return the center square of the frame


# define a function to load a video and return its frames as a numpy array
def load_the_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path) # read the video at the given path
    frames = []
    try:
        while True:
            ret, frame = cap.read() # read the next frame
            if not ret:
                break
            frame = crop_square(frame) # crop the center square of the frame
            frame = cv2.resize(frame, resize) # resize the frame
            frame = frame[:, :, [2, 1, 0]] # swap the color channels
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) # return the list of frames as a numpy array

def build_feature_extractor():
    # Load pre-trained InceptionV3 model with pre-trained ImageNet weights
    features_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # Define a function to preprocess the input image using InceptionV3 preprocessing function
    preprocess_input = keras.applications.inception_v3.preprocess_input

    # Define input tensor
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    #preprocess the tensor
    preprocessed = preprocess_input(inputs)

    # Pass preprocessed input through the feature extractor model
    outputs = features_extractor(preprocessed)
    # Pass preprocessed input through the feature extractor model
    return keras.Model(inputs, outputs, name="features_extractor")


# Build a feature extractor model and store it in features_extractor variable
features_extractor = build_feature_extractor()

# Define a function to prepare all the videos in the given dataframe
def load_all_videos(df, root_dir):
    # Get the number of videos in the dataframe
    num_samples = len(df)
    # Get the paths of all the videos in the dataframe
    video_paths = list(df.index)
    labels = df["label"].values
    labels = np.array(labels=='FAKE').astype(np.int)

    
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # loop through each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_the_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Create placeholders to store the masks and features of the current video
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            # Loop through each frame of the current video and extract the features for each frame.
            for j in range(length):
                temp_frame_features[i, j, :] = features_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        # Assign the extracted features and masks of the current video
        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

from sklearn.model_selection import train_test_split

#Split the sample_train_metadata dataframe into train and test sets using the train_test_split function
Train_set, Test_set = train_test_split(sample_train_metadata,test_size=0.1,random_state=42,stratify=sample_train_metadata['label'])

print(Train_set.shape, Test_set.shape )

#prepare train set and test set by extracting frame features
train_data, train_labels = load_all_videos(Train_set, "train")
test_data, test_labels = load_all_videos(Test_set, "test")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")
