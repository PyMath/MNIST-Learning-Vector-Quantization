# MNIST-Learning-Vector-Quantization
Explore the effects of prototype selection on the popular MNIST dataset.  

## Overview
Learning Vector Quantization is a method where initial points are chosen from within a dataset and called the prototypes. There are various ways on actually obtaining this initialization such as k-means clustering and taking the initial centroids as the starting points. These points are then 'trained' by evaluating them against the training set by:  
1) Starting with a training image and then selecting the point closest in distance (Euclidean metric used here)
2) Comparing labels of the prototype with labels of the training image.  
3) If the labels match then move the prototype point closer to the current image scaled by some learning rate or move it further from the selected point, also scaled by some learning factor
4) Repeat this process through all the training data for a desired number of epochs  
## Usage
1. Make sure all dependencies are met. This implementation uses NumPy and SkLearn
```
pip install -r requirements.txt
```
2. Run main script
```
python lbq.py -np=int -nc=int -lr=float -e=int -nt=int -mc=float -save=str
```
-np, --num_protos: Number of prototypes to initialize  
-nc, --num_classes: Number of classes associated with dataset (script assumes all classes used in MNIST)  
-lr, --learning_rate: Learning rate to train the prototypes  
-e, --epochs: how many times to go over training set  
-nt, --num_train: how many times to perform training process  
-mc, --mean_confidence: degree for mean confidence interval  
-save: File path to save best trained prototypes  
