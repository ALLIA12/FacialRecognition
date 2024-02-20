# Multi-Task Learning Model for Face Detection and Recognition

This project develops a multi-task Convolutional Neural Network (CNN) using PyTorch for simultaneous face detection and
identity recognition on the CelebA dataset. It also includes a real-time face detection application using webcam feed.

## Installation

Before running the project, ensure you have Python 3.8 or later installed. Follow the steps below to set up your
environment:

### Clone the Repository

```bash
git clone https://github.com/ALLIA12/FacialRecognition.git
cd FacialRecognition
```

### Install Dependencies

- PyTorch, torchvision, and torchaudio for deep learning models.
- pandas and pyarrow for data manipulation.
- opencv-python for image processing.
- Install the required packages using pip:

```bash
pip install torch torchvision torchaudio pandas pyarrow opencv-python
```

## Dataset

The CelebA dataset is used, which needs to be downloaded and organized in the specified directory structure. Ensure the
following files are placed correctly:

- img_celeba/img_celeba: Directory containing CelebA images.

- img_celeba:
    - list_bbox_celeba.txt: Bounding boxes for faces.
    - identity_CelebA.txt: Identity for each face.
    - list_eval_partition.txt: Data partition file (training, validation, test splits).

## Usage

Training the Model
Run the first modelTorch notebook to train the multi-task learning model. This notebook will:

- Preprocess and load the CelebA dataset.
- Define and train the MultiTaskCNN model for both bounding box regression and identity classification.
- Evaluate the model on the validation and test sets.
- Save the trained model for later use.

## Real-Time Face Detection

After training, use the predication Python script to start the real-time face detection application. This script will:

- Load the trained model.
- Capture video from the webcam.
- Detect faces and their identities in real-time.
- Display the webcam feed with bounding boxes drawn around detected faces.

Ensure the trained model file multi_task_cnn_model.pth is located in the PyTorchModel directory.

## Technical Details

- Model Architecture: Utilizes ResNet-18 as a feature extractor with two heads for predicting bounding boxes and
  classifying identities.
    - The bounding boxes are correctly identified, but the identity prediction leaves much to be desired
    - Dataset: CelebA dataset for training and validation.
    - Real-Time Detection: Implemented using OpenCV and Tkinter for GUI.

- For training I recommend using WSL, since it is the easiest one to set up
- For Predication in real time, use windows since it is easier to access the camera there