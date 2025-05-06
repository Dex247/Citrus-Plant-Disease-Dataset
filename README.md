# A README file to explain the purpose of the repository, the dataset, and the code on the manuscript titled “Enhanced Orange Fruit Disease Detection Via An Adamax-Optimized Yolov11 Nano In Precision Agriculture”

The Choice of using YOLOv11 Nano
The choice of YOLOv11 Nano for this study was driven by its remarkable balance of accuracy, speed, and efficiency. Traditional disease detection methods often require extensive resources and sophisticated equipment that are not feasible for medium-scale farms. The YOLOv11 Nano, however, offers a transformative approach due to its lightweight architecture specifically designed to perform real-time object detection even in resource-limited settings.
One of the primary strengths of YOLOv11 Nano is its streamlined convolutional network, which allows for rapid processing of input data. This is critical for early-stage disease detection in orange fruits where timely identification can prevent the spread and severity of infections. The model's efficiency ensures that it can be deployed on edge devices like smartphones and drones, which are increasingly accessible to farmers. This allows for immediate, on-the-spot analysis in the field, reducing delays associated with traditional lab-based diagnostics.

Problem Statement
The manuscript begins by establishing the importance of disease detection in oranges to mitigate crop loss and improve productivity. Disease identification, particularly in resource-constrained environments, requires models that are:
- Lightweight and computationally efficient.
- Accurate in detection.
- Compatible with low-power devices.
The problem statement emphasizes the challenges in deploying traditional deep learning models in agricultural settings due to their computational and hardware demands.

Model Architecture
The manuscript introduces the YOLOv11 Nano model, highlighting its lightweight design and adaptability:
- Depthwise Convolution Layers: Incorporated to reduce model complexity, achieving a 4.2× reduction in FLOPs (floating-point operations per second) and a 68% smaller model size compared to traditional approaches.
-  Optimization Strategy: Replacing the default Adam optimizer with Adamax (a variant leveraging the infinity norm) to handle sparse gradients more effectively.
- Dynamic Parameter Adjustment: A novel dynamic optimization strategy was implemented to adaptively modify the learning rate, beta1, and beta2 parameters during training, enhancing convergence stability.


Dataset Curation
The manuscript details the creation of a dataset specifically tailored for the study:
- Scope: A curated dataset of 700 annotated images of diseased and healthy oranges was used, covering conditions like black spots, cankers, greening and scab — prevalent diseases affecting orange fruits. The dataset was obtained from the Mendeley dataset repository.
- Annotation Process: Annotation was performed using bounding boxes to mark regions of interest, ensuring high-quality labels for training and evaluation. Annotation was done using Cvat annotation tool. The annotated files were then converted into a Yolo format. 
-Diversity: The dataset includes images captured under varying lighting conditions and angles to simulate real-world scenarios.


Experimental Results
The results section of the manuscript focuses on the performance metrics and comparative analysis:
- Mean Average Precision (mAP): The Adamax-optimized YOLOv11 Nano achieved **95% mAP accuracy**, surpassing baseline models in precision.
- Inference Latency: A 21% reduction in inference latency compared to YOLOv8 Nano is reported, demonstrating the model's suitability for real-time applications.
- Resource Efficiency: The model requires only **1.8 GB RAM**, ensuring compatibility with low-power devices commonly available in agricultural environments.
  

Training on YOLOv11 
Prerequisites
Server Setup: NVIDIA GeForce RTX 3090 graphics card, which is highly compatible with YOLOv11. The YOLOv11 is compatible with CUDA and the CuDNN and features a strong parallel processing architecture.   Having 62.5 GB of RAM and 10 GB of hard drive space allows our system to quickly load models and handle massive amounts of data. Ubuntu 20.04.5 LTS (GNU/Linux 5.15.0-76-generic x86_64)
Python Environment: Python -3.9.12, torch-2.5.0+cu118, Ultralytics – 8.3.23
YOLOv9 Source Code: Obtain the YOLOv9 implementation.

Setting up the python environment:
sudo apt update
sudo apt install python3-venv  # Installing python3-venv if not installed
python3 -m venv yolov11_custom-env
source yolov11_custom-env/bin/activate
pip install --upgrade pip
pip install torch torchvision opencv-python

Cloning the YOLOv11 Repository from github
git clone https://github.com/ultralytics/yolov11.git
cd yolov11
pip install -r requirements.txt

Creating directory and uploading the dataset via SFTP
Using the Termius's SFTP client to upload the dataset to the server. ../home/ben/Yolov11_Custom
mkdir -p datasets/ home/ben/Yolov11_Custom /images
mkdir -p datasets/ home/ben/Yolov11_Custom /val

# images and label files were uploaded to respective directories

Training the Model
Parameter adjustments
Downloading Yolov11nano from github


Introducing the Adamax Optimizer
Adamax is an optimizer that builds upon the Adam optimization algorithm. It uses the infinity norm (maximum) of the gradient instead of the L2 norm. This provides superior stability and robustness against sparse gradients, making it suitable for tasks with noisy or irregular updates.
The Adamax optimizer in YOLOv11 was modified in the training script and configuration files.


from ultralytics import YOLO


#model = YOLO("yolo11.yaml")
model = YOLO("best.pt")

#model.train(data = "dataset_custom.yaml", imgsz = 640, batch = 8, epochs = 300, workers = 8, device = 0, optimizer="Adamax")

model.val(
    data="dataset_custom.yaml",
    imgsz=640,
    batch=8,
    device=0,          # GPU 0
    workers=8,       # Use 'test' for test set evaluation
)


The parameters for the Adamax optimizer, such as learning rate, beta1, and beta2 were modified in the configuration python file. Other parameters such as batch, epochs, were continuously modified.


import torch
import torch.optim as optim

optimizer = optim.Adamax(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

lr: Learning rate (default is 0.002, but the learning rate was continuously adjusted for an enhanced model accuracy and performance).
betas: Coefficients used for computing running averages of gradient (beta1) and its square (beta2).

for epoch in range(num_epochs):
    for data, target in train_loader:  # using a DataLoader
        optimizer.zero_grad()          # Clearing existing gradients
        output = model(data)           # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()                # Backward pass
        optimizer.step()               # Update weights


Implementing a Dynamic Adjustments

for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.95  # Reducing and adjusting the learning rate by 5%


Depthwise Convolution Layer

Importing Necessary Modules

import torch
import torch.nn as nn

depthwise_conv = nn.Conv2d(
    in_channels=32,  # Number of input channels
    out_channels=32, # Number of output channels (same as in_channels for depthwise)
    kernel_size=3,   # Kernel size
    stride=1,        # Stride
    padding=1,       # Padding
    groups=32        # Set groups equal to in_channels for depthwise convolution
)


Integratting the depthwise convolutions into the model

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

The Standard Convolution in YOLOv11 was replaced with the DepthwiseSeperableConv
The standard convolution layers in YOLOv11 model was replaced with a DepthwiseSeparableConv where computational efficiency is critical. This approach enhanced and optimized the model.
