# ResNet Project

This repository contains an implementation of the ResNet model for image classification using PyTorch. The model is trained on the CIFAR-10 dataset under [5 million parameters](https://github.com/sidakwalia/DL-Project/blob/main/save_model/Best_model_parameters.txt).

Best Accuracy: 94.72% on test dataset


Parameters: 4,959,770

# Project Structure
| Directory / File | Description |
|-----------------|-----------------|
| best_models | Contains models tuned on various hypermaters | 
| models | Resnet definitions and model configuartion | 
| results | Contains the logs and plots |
| save_model | Contains the weights of the best model and its parameters logs |
| job.SBATCH | runs train.py |
| requirements.txt | Libraries needed to run the model |
| test.py | test model |
| train.py | train model |

Table of Contents
Getting Started
Prerequisites
Installation
Usage
Results
Model Architecture
License
Acknowledgments
References
1. Create a virtual conda environment :conda create --name resnet 
2. Activate the conda environment  :conda activate resnet
3. pip3 install -r requirements.txt :
3. Run the python3 train.py
4. Run the python3 test.py


**Results**

This [trained model](https://github.com/sidakwalia/DL-Project/blob/main/best_models/sgd_0.001.pt) achieved an accuracy of 94.72% on the CIFAR-10 test set. The model can be further optimized by adjusting hyperparameters and using data augmentation techniques.

| ![Train and Test Accuracy](https://user-images.githubusercontent.com/25876670/232180526-ac2bd921-18d0-4b1c-9bd3-e426e9a3a069.png) | 
|:--:| 
| *Train and Test Accuracy* |

**Model Architecture**

The ResNet model architecture used in this project is based on the ResNet paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

The architecture consists of multiple layers, including convolutional layers, residual blocks, and fully connected layers.

**License**

This project is licensed under the MIT License - see the LICENSE file for details.

**References**

[PyTorch documentation](https://pytorch.org/docs/stable/index.html)

CIFAR-10 dataset
