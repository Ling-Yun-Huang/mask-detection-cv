"""
File: 4_test_function_image.py
Project: Mask-Wear Detection using Computer Vision Techniques
Author: Ling-Yun Huang
Description: This script tests the trained model with image inputs.
"""

# install pre-trained models of MTCNN for PyTorch
get_ipython().system('pip install facenet-pytorch')


## Library import

# for accessing and preprocessing the video dataset
import cv2
import numpy as np
from skimage import img_as_ubyte, io, color
import matplotlib.pyplot as plt

# face detector
from facenet_pytorch import MTCNN

# for animation the video
from matplotlib import rc
import matplotlib.animation as animation

# Images output with rectangle and label
from matplotlib import patches

# CNNs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

get_ipython().run_line_magic('matplotlib', 'inline')


## MaskDetectionVideo Function Prepare

# Video loaded
def video_loaded(video_dir):

    cap = cv2.VideoCapture(video_dir)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while fc < frameCount and ret:
        ret, video[fc] = cap.read()
        video[fc] = cv2.cvtColor(video[fc], cv2.COLOR_BGR2RGB)
        fc += 1

    cap.release()

    return video

# Define a dictionary to map predicted labels to text
label_text = { '0': "No mask", '1': "Mask", '2': "Mask incorrect"}

### CNN

## CNNs architecture
# CNNs with one layer of conventional layer
class CNN_1(nn.Module):
    def __init__(self, hidden_size, kernel_size):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_size,
                               kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(hidden_size * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CNNs with two layers of conventional layers
class CNN_2(nn.Module):
    def __init__(self, hidden_size, kernel_size):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_size,
                               kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size,
                               kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(hidden_size * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CNN
def CNN_image(classifier, image):
    data_means = [0.485, 0.456, 0.406]
    data_stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_means, std=data_stds)
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predicted the image
    classifier.eval()
    with torch.no_grad():
        output = classifier(img_tensor)
        _, predicted_label = torch.max(output, 1)
        predicted_label = predicted_label.item()  # Convert to Python scalar

    label = label_text.get(str(predicted_label), "Unknown")

    return label


## The face detector function *The MTCNN pre-trained model code are adopted from Lab 8.*

def MTCNN_predicted_image(img, ax):

    mtcnn = MTCNN(keep_all=True)
    faces_MTCNN, _ = mtcnn.detect(img, landmarks=False)

    if faces_MTCNN is not None:
        for face in faces_MTCNN:
            x, y, w, h = face.astype(int)

            if w > 0 and h > 0:
                face_test = img[y:h, x:w]  # Extract the detected face region from the original image
                resized_image = cv2.resize(face_test, (32, 32)) # resize into 32x32
                normalized_image = resized_image.astype('float32') / 255.0 # Convert to float and normalize

                CNN_classifier = torch.load(os.path.join(GOOGLE_DRIVE_PATH, 'Models/CNN_classifier.pth'))
                label = CNN_image(CNN_classifier, normalized_image)

                # Add rectangle and predicted label on image
                ax.add_patch(patches.Rectangle(xy=(face[0], face[1]), width=face[2]-face[0], height=face[3]-face[1],
                                    fill=False, color='r', linewidth=1.5, label=label))
                ax.text(face[0], face[1] - 5, label, color='r', fontsize=10)


## Animate video with predictions

# animate video with prediction label
def animate_video(video):
    rc('animation', html='jshtml')

    fig, ax = plt.subplots(figsize=(5, 3))

    def frame(i):
        ax.clear()
        ax.axis('off')
        fig.tight_layout()

        # Apply MTCNN_predictedCNN function to the ith frame
        MTCNN_predicted_image(video[i*10+100, :, :, :], ax)

        # Plot the frame
        plot=ax.imshow(video[i*10+100, :, :, :])
        return plot


    anim = animation.FuncAnimation(fig, frame, frames=80)
    plt.close()

    return anim


## MaskDetectionVideo Function

# ***Note that the "PATH" must the direction of the folder that have all models and dataset***

def MaskDetectionVideo(path_to_video):

    video = video_loaded(path_to_video) # loaded video
    anim = animate_video(video) # animate the video with predict label

    return anim

# define video path
video_dir = os.path.join(PATH, 'Video/Video.mp4')

# Load the CNN model
CNN_classifier = torch.load(os.path.join(PATH, 'Models/CNN_classifier.pth'))

## For running the result
# MaskDetectionVideo(video_dir)

