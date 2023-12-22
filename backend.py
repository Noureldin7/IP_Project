import numpy as np
import time
import cv2
import pickle
from glob import glob
from sklearnex import patch_sklearn
patch_sklearn()
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

def resize_and_pad(img, target_width = 128, target_height = 128):
    # Get the original image dimensions
    height, width, *_ = img.shape

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Determine the new dimensions to fit into a 128x128 frame
    if aspect_ratio > target_width/target_height:  # Landscape orientation
        new_width = target_width
        new_height = int(target_height / aspect_ratio)
    else:  # Portrait or square orientation
        new_width = int(target_width * aspect_ratio)
        new_height = target_height

    # Resize the image while maintaining the original aspect ratio
    resized_image = cv2.resize(img, (new_width, new_height))

    # Calculate the amount of padding needed
    x_padding = (target_width - new_width) // 2
    y_padding = (target_height - new_height) // 2

    # Add zero-padding to the resized image
    padded_image = cv2.copyMakeBorder(resized_image, y_padding, y_padding, x_padding, x_padding, cv2.BORDER_CONSTANT, value=0)
    if padded_image.shape != (target_height, target_width):
        padded_image = cv2.resize(padded_image, (target_width, target_height))
    return padded_image

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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SE, iterations=15)
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(ctrs[0][0])
    modified = np.array(img)
    if len(ctrs) != 0:
        x, y, w, h = cv2.boundingRect(max(ctrs, key=cv2.contourArea))
        cv2.rectangle(modified, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (original_img_shape[:2]))
        # img = resize_and_pad(img)
    # img[mask == 0] = 0
    h = process_HOG(img).reshape(-1)
    return h, modified
def confidence_threshold(row, threshold):
	max_confidence = np.max(row)
	if max_confidence < threshold:
		return -1
	else:
		return np.argmax(row)

model = pickle.load(open("./model.pkl", "rb"))
PCA_model = pickle.load(open("./pca.pkl", "rb"))

def predict(frame):
    features, modified = preprocess(frame)
    # print(modified.shape)
    # cv2.imshow('Frame', modified)
    features = PCA_model.transform([features])
    confidence = model.predict_proba(features)
    # prediction = model.predict(features)
    prediction = confidence_threshold(confidence, 0.6)
    return prediction

#! Call Rock Dislike Fist Palm
if __name__ == "__main__":
    # Open the camera (default camera index is usually 0)
    cap = cv2.VideoCapture(0)
    # load model from pickle file
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
            features, modified = preprocess(frame)
            # print(modified.shape)
            cv2.imshow('Frame', modified)
            features = PCA_model.transform([features])
            confidence = model.predict_proba(features)
            # prediction = model.predict(features)
            prediction = confidence_threshold(confidence, 0.6)
            print(prediction)
            # Wait for 1 second (1000 milliseconds)
            time.sleep(0.5)

            # Break the loop if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()