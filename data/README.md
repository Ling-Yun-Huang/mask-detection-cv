# Data Folder

This folder provides detailed information about the datasets used in the **Mask-Wearing Detection with Computer Vision Techniques** project.

## Data Overview

The project uses two main types of datasets: **image datasets** (split into training and testing sets) and **video data**.

Below is a detailed description:

### 1. Image Datasets
The image data is split into two subsets:

#### a) Training Dataset
- **Description**:  
  This dataset is used to train the machine learning models. It contains labeled images categorized into three classes:  
  - **No Mask**: 376 samples  
  - **Mask Proper Wear**: 1940 samples  
  - **Mask Without Proper Wear**: 48 samples  
- **Challenges**:  
  The dataset is imbalanced, requiring techniques such as oversampling, undersampling, or class weighting during model training.

#### b) Testing Dataset
- **Description**:  
  This dataset is held out during training and is only used to evaluate the performance of each implemented model.  
- **Note**:  
  The testing dataset remains unseen throughout the training phase to ensure unbiased evaluation.

- **Availability of Image Datasets**:  
  Due to copyright and licensing restrictions, the image datasets are **not included** in this repository.

### 2. Video Dataset
- **Description**:  
  The video, sourced from the **Taiwan Centres for Disease Control and Prevention**, was used as a supplementary resource to test the models’ predictive abilities. It includes scenarios where individuals:  
  - Are not wearing masks.  
  - Are wearing masks correctly.  
  - Are wearing masks incorrectly.  
- **Source**:
  - Taiwan CDC Communication Resources: [Website](https://www.cdc.gov.tw/Advocacy/SubIndex/2xHloQ6fXNagOKPnayrjgQ?diseaseId=N6XvFa1YP9CXYdB0kNSA9A).
  - Direct Video Link: [Download here](https://www.cdc.gov.tw/File/Get/IgN4Nnj0UhtIPNRMED64ew).
- **Licensing**:  
  - Health education materials from the Taiwan CDC may be quoted for non-profit purposes within a reasonable scope.  
  - Ensure to cite the source when quoting.
- **Availability**:  
  - The original video file is **not included in this repository**. Please refer to the Taiwan CDC website for access.

---

## Notes
- The datasets described above are not included in this repository.  
- To replicate this project, you can source similar datasets or contact the original providers for access.  
- Ensure compliance with all applicable licenses and copyright laws when using data for your own research or projects.

## Citation
If you reference the video data, please cite:  
> Taiwan Centres for Disease Control and Prevention. [Advocacy Materials](https://www.cdc.gov.tw/Advocacy/SubIndex/2xHloQ6fXNagOKPnayrjgQ?diseaseId=N6XvFa1YP9CXYdB0kNSA9A).
