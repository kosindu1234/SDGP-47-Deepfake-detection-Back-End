#pip install tensorflow
#pip install matplotlib
#pip install imageio
#pip install pandas

#!pip install -U --upgrade tensorflow


#!pip install tensorflow
def detect():


    from keras.models import load_model

    new_model = load_model(r'C:\Users\nimasha\SDGP-47-Deepfake-detection-Back-End\model.h5') #model path
    new_model.summary()

    from tensorflow import keras
    #from imutils import paths
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import imageio
    import cv2
    import os
    #from IPython.display import HTML
    from base64 import b64encode

    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 10

    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 2048

    DATASET_FOLDER = '../input/deepfake-detection-challenge'
    sample_train_folder = 'train_sample_videos'
    TEST_FOLDER = 'test_videos'

    #test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATASET_FOLDER, TEST_FOLDER))), columns=['video'])

    #Define a function to play a video in the notebook
    def play_the_video(video_file, subset=sample_train_folder):
        # Read the contents of the video file
        video_url = open(os.path.join(DATASET_FOLDER, subset,video_file),'rb').read()
        # Encode the video content in base64 format
        data_url = "data:video/mp4;base64," + b64encode(video_url).decode()
        # Create an HTML object that contains a video element with the specified source URL
        return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)


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

    # define a function to crop the center square of a given frame
    def crop_square(frame):
        y, x = frame.shape[0:2] # get the height and width of the frame
        min_dim = min(y, x) # get the minimum of the two dimensions
        start_x_axis = (x // 2) - (min_dim // 2) # calculate the starting point for x-axis
        start_y_axis = (y // 2) - (min_dim // 2) # calculate the starting point for y-axis
        return frame[start_y_axis : start_y_axis + min_dim, start_x_axis : start_x_axis + min_dim] # re
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

    def prepare_predict_video(frames):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                frame_features[i, j, :] = features_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_mask

    def prediction(path):
        frames = load_the_video(os.path.join(DATASET_FOLDER, TEST_FOLDER,path))
        frame_features, frame_mask = prepare_predict_video(frames)
        return new_model.predict([frame_features, frame_mask])[0]
        

    def convert_gif(images):
        converted_images = images.astype(np.uint8)
        imageio.mimsave("animation.gif", converted_images, fps=10)
        return embed.embed_file("animation.gif")


    test_video = (r"C:\Users\nimasha\SDGP-47-Deepfake-detection-Back-End\fake video tom.mp4")#video path
    print(f"Test video path: {test_video}")
    pred_probs = prediction(test_video)
    print(f"Predicted class probabilities: {pred_probs}")
    if(prediction(test_video)>=0.55):
        print(f'The predicted class of the video is FAKE')
    else:
        print(f'The predicted class of the video is REAL')

    play_the_video(test_video,TEST_FOLDER)

detect()
