This repository contains an implementation of the ResNet model for image classification using PyTorch. The model is trained on the CIFAR-10 dataset and achieves an accuracy of 90% on the test set.

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


Results
The trained model achieves an accuracy of 90% on the CIFAR-10 test set. The model can be further optimized by adjusting hyperparameters and using data augmentation techniques.

Model Architecture
The ResNet model architecture used in this project is based on the ResNet paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

The architecture consists of multiple layers, including convolutional layers, residual blocks, and fully connected layers.


License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This implementation is based on the ResNet paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

References
PyTorch documentation
CIFAR-10 dataset