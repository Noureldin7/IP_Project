import numpy as np
import time
import os
import cv2
import pickle
from glob import glob
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from IPython.display import display_html
from itertools import chain,cycle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern, hog
import concurrent.futures # for multithreading
def do_parallel_work(function : callable, *function_arguments : list[any]) -> list[any]:
    """Run a function in parallel on multiple threads with the given arguments.\n
    Notice that the function must be thread-safe.\n
    Notice that the function arguments are passed as a list of each argument not a list of all arguments.\n
    e.g.:\n
        do_parallel_work(function, [1, 2, 3], [4, 5, 6])\n
        executes the function with the arguments:\n
        function(1, 4)\n
        function(2, 5)\n
        function(3, 6)\n
    Returns a list of the function's return values.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(function, *function_arguments))

def wb(channel, perc = 5):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi+0.01), 0, 255))
    return channel

def process_YCrCb(img):
    
    yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    lower_color = np.array([80, 140, 115])
    upper_color = np.array([255, 160, 135])
    color_mask = cv2.inRange(yCrCb, lower_color, upper_color)
    
    return color_mask

def process_chroma_balance(img):
    yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    yCrCb[:,:,1] = wb(yCrCb[:, :, 1])
    yCrCb = cv2.cvtColor(yCrCb, cv2.COLOR_YCrCb2BGR)
    yCrCb[np.argmax(yCrCb, axis=2) == 0] = 0
    yCrCb[np.argmax(yCrCb, axis=2) == 1] = 0
    yCrCb[np.argmax(yCrCb, axis=2) == 2] = 255

    return yCrCb

def process_Canny(img):
    grey = np.array(img)
    if len(img.shape)==3:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 0, 0.3, apertureSize=7)
    
    return edges

def process_Laplacian(img):
    grey = np.array(img)
    if len(img.shape)==3:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(grey, -1, ksize=5)
    
    return edges

def process_Sobel(img):
    grey = np.array(img)
    if len(img.shape)==3:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(grey, -1, 1, 1)
    
    return edges


def process_HOG(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = hog(gray, orientations=8, pixels_per_cell=(16, 16))
    # hog = cv2.HOGDescriptor()
    # h = hog.compute(img, winStride=(32,32))
    return h

def skin_dumb(img):
    img_rgb = np.array(img)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    clf = pickle.load(open("./decision_tree.pkl", "rb"))
    img_len = img_rgb.shape[0] * img_rgb.shape[1]
    X = np.zeros((img_len, 14))

    X[:, 0] = img_rgb[:, :, 0].reshape(-1)
    X[:, 1] = img_rgb[:, :, 1].reshape(-1)
    X[:, 2] = img_rgb[:, :, 2].reshape(-1)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    X[:, 3] = img_hsv[:, :, 0].reshape(-1)
    X[:, 4] = img_hsv[:, :, 1].reshape(-1)
    X[:, 5] = img_hsv[:, :, 2].reshape(-1)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    X[:, 6] = img_lab[:, :, 0].reshape(-1)
    X[:, 7] = img_lab[:, :, 1].reshape(-1)
    X[:, 8] = img_lab[:, :, 2].reshape(-1)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    X[:, 9] = img_gray.reshape(-1)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    X[:, 10] = img_ycrcb[:, :, 0].reshape(-1)
    X[:, 11] = img_ycrcb[:, :, 1].reshape(-1) 
    X[:, 12] = img_ycrcb[:, :, 2].reshape(-1)
    img_lbp = local_binary_pattern(img_gray, 8, 1, method='uniform')
    X[:, 13] = img_lbp.reshape(-1)

    mask = clf.predict(X).reshape(img_rgb.shape[0], img_rgb.shape[1])

    img_rgb[mask == 0] = 0

    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return mask

def process_MeanShift(img, sp, sr):
    rgb_img = np.array(img)
    return cv2.pyrMeanShiftFiltering(rgb_img, sp, sr)

# Build Preprocessing Pipeline Here
def preprocess(img):
    img = cv2.resize(img, (128, 128))
    original_img_shape = img.shape
    img = cv2.GaussianBlur(img, (3, 3), 8)
    mask = skin_dumb(img)
    mask = np.uint8(mask) * 255
    SE = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SE, iterations=9)
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(ctrs[0][0])
    if len(ctrs) != 0:
        x, y, w, h = cv2.boundingRect(max(ctrs, key=cv2.contourArea))
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (original_img_shape[:2]))
    # img[mask == 0] = 0
    h = process_HOG(img).reshape(-1)
    return h
def confidence_threshold(row, threshold):
	max_confidence = np.max(row)
	if max_confidence < threshold:
		return -1
	else:
		return np.argmax(row)


if __name__ == "__main__":
    # Open the camera (default camera index is usually 0)
    cap = cv2.VideoCapture(0)
    # load model from pickle file
    model = pickle.load(open("./model.pkl", "rb"))
    # model.probability = True
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Check if the frame was successfully read
            if not ret:
                print("Error: Could not read frame.")
                break

            # Display the frame (you can remove this line if you don't want to display the frames)
            cv2.imshow('Frame', frame)
            features = preprocess(frame)
            PCA_model = pickle.load(open("./PCA.pkl", "rb"))
            features = PCA_model.transform([features])
            prediction = model.predict(features)
            # confidence = model.predict_proba(features)
            # prediction = confidence_threshold(confidence, 0.45)
            print(prediction)
            # Wait for 1 second (1000 milliseconds)
            time.sleep(1)

            # Break the loop if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()