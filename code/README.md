# Mask-Wear Detection using Computer Vision Techniques - Code Overview

This folder contains the code for the **Mask-Wear Detection** project, which uses machine learning and computer vision techniques to detect mask-wearing status. The project is broken down into multiple parts for easier management and execution.

## File Structure

#### 1. Model Training and Testing
- **1_train_model_image.ipynb**:  
  Jupyter notebook for training machine learning models on the image training dataset. This notebook includes:
  - Data preprocessing
  - Model selection (e.g., CNN, SVM, MLP)
  - Training the models using the training data.
  
- **2_test_model_image.ipynb**:  
  Jupyter notebook for testing the trained models on the image testing dataset. This notebook includes:
  - Performance evaluation (e.g., accuracy, precision, recall)
  - Confusion matrix and model metrics for assessing the model's performance on the test dataset.
  
- **3_test_model_video.ipynb**:  
  Jupyter notebook for testing the trained models on video data. It processes the video frames to detect the mask-wearing status and includes:
  - Evaluating the model's performance on video frames
  - Generating predictions for mask-wearing detection in video format.
  
#### 2. Testing Functions (Python Scripts)
- **4_test_function_image.py**:  
  Python script that defines a function for testing the model's performance on the image dataset. It takes an image input, makes predictions using the trained model, and outputs performance metrics.
  
- **5_test_function_video.py**:  
  Python script that defines a function for testing the model's performance on video data. The function processes video frames, makes predictions, and outputs performance metrics for video testing.

## Data Requirements
- **Image Dataset**:  
  The image dataset includes training and testing subsets. **It is not included** in this repository due to privacy and copyright restrictions. You need to ensure the following:
  - Training set (with labeled images for mask-wearing status)
  - Testing set (for evaluating the model's performance)
  
- **Video Dataset**:  
  The video dataset is used for testing the trained models. It is sourced from the **Taiwan Centres for Disease Control and Prevention**. **The video is not included** in this repository. Please ensure you have access to the video for testing purposes:
  - [Download the video from Taiwan CDC](https://www.cdc.gov.tw/Advocacy/SubIndex/2xHloQ6fXNagOKPnayrjgQ?diseaseId=N6XvFa1YP9CXYdB0kNSA9A)
  
## Installation Requirements
To run the code, you will need the libraries listed in the `requirements.txt` file. You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```
---

# Usage

To run the project:

### Training:
Run `1_train_model_image.ipynb` to train the model with the image training dataset.

### Testing (Image):
Run `2_test_model_image.ipynb` to test the trained model with the image testing dataset.

### Testing (Video):
Run `3_test_model_video.ipynb` to test the model with the video data.

### Testing with Functions:
You can also use standalone functions to test the model:

- **Test Function (Image)**:  
  Use `4_test_function_image.py` to test the model with an image input. This script can be integrated into other systems for mask-wearing detection.

- **Test Function (Video)**:  
  Use `5_test_function_video.py` to test the model with video input. The script processes video frames and returns mask-wearing predictions.

---

# Licensing
The code is licensed under the [MIT License](../LICENSE).

---

# Notes
- Ensure the image and video datasets are properly referenced in the code for successful execution.
- You can modify the code to support other types of data as long as they are structured similarly to the image and video datasets.
- The models and code are designed for **educational** and **non-commercial** purposes.
