from PIL import ImageTk, ImageDraw
import cv2
import pandas as pd

import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import tkinter as tk


class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiTaskCNN, self).__init__()
        # Load a pre-trained model as a feature extractor
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-2])  # Exclude the last fc layer

        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, 4)  # 4 outputs for bbox (x, y, width, height)
        )

        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        bbox = self.bbox_head(x)
        class_logits = self.classifier_head(x)
        return bbox, class_logits


def load_model(file_name, num_classes, directory='PyTorchModel'):
    path = os.path.join(directory, file_name)
    # Initialize the model
    model = MultiTaskCNN(num_classes=num_classes)
    # Load the model state dictionary
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
base_dir = 'img_celeba'
bbox_excel_path = os.path.join(base_dir, 'list_bbox_celeba.txt')
identity_txt_path = os.path.join(base_dir, 'identity_CelebA.txt')
images_dir = os.path.join(base_dir, 'img_celeba')

# Read the Excel file for bounding boxes
bbox_df = pd.read_csv(bbox_excel_path, sep='\s+', skiprows=1)

# Read the identity file
identity_df = pd.read_csv(identity_txt_path, sep=" ", header=None, names=['image_id', 'identity'])
# Path to the evaluation partition file
eval_partition_path = os.path.join(base_dir, 'list_eval_partition.txt')

# Read the evaluation partition file
eval_partition_df = pd.read_csv(eval_partition_path, sep='\s+',
                                header=None, names=['image_id', 'evaluation_status'])

# Merge the evaluation partition data with the bounding boxes and identity data
merged_df = pd.merge(bbox_df, identity_df, on='image_id')
merged_df = pd.merge(merged_df, eval_partition_df, on='image_id')

# Split the merged data into training, validation, and testing datasets
train_df = merged_df[merged_df['evaluation_status'] == 0].drop(columns=['evaluation_status'])
val_df = merged_df[merged_df['evaluation_status'] == 1].drop(columns=['evaluation_status'])
test_df = merged_df[merged_df['evaluation_status'] == 2].drop(columns=['evaluation_status'])

num_classes = len(train_df['identity'].unique())
file_name = 'multi_task_cnn_model.pth'
model = load_model(file_name, num_classes)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize camera
cap = cv2.VideoCapture(0)


def update():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        window.quit()
        return

    # Resize the frame to 128x128 for the model
    resized_frame = cv2.resize(frame, (128, 128))

    # Convert the image to RGB (OpenCV uses BGR)
    rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(rgb_image)

    # Apply the transformations
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        bbox, class_logits = model(input_batch)

    # Scale the bbox back up to the frame's original size
    scale_x = frame.shape[1] / 128
    scale_y = frame.shape[0] / 128
    bbox_scaled = (bbox.cpu().numpy()[0] * [scale_x, scale_y, scale_x, scale_y]).astype(int)

    # Draw the bounding box on the original frame (not resized)
    pil_image_original = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image_original)
    draw.rectangle(
        [(bbox_scaled[0], bbox_scaled[1]), (bbox_scaled[0] + bbox_scaled[2], bbox_scaled[1] + bbox_scaled[3])],
        outline="red")

    # Convert PIL Image back to a format that can be displayed by Tkinter
    tk_image = ImageTk.PhotoImage(pil_image_original)

    # Update the label with the new image
    label.imgtk = tk_image
    label.configure(image=tk_image)
    label.after(10, update)  # Schedule the next update


window = tk.Tk()
window.title("MultiTaskCNN Inference")

label = tk.Label(window)
label.pack()

update()  # Start the frame updates

window.mainloop()

cap.release()
