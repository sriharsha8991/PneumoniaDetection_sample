# Pneumonia Detection Model using X-ray Images
------New change----

## Introduction
This project focuses on detecting pneumonia using X-ray images, employing Convolutional Neural Networks (CNNs) and transfer learning. The development process we used in this project is outlined below, highlighting the rationale and choices made during each step.

## 1. Dataset Acquisition
The dataset is obtained from an Amazon S3 bucket using the `wget` command, ensuring reliable and scalable access.

## 2. Module Import and Dataset Extraction
Relevant modules, including TensorFlow and data visualization tools, are imported. The dataset is extracted into a new folder ('xray_dataset') for easy access.
Stored the xray_dataset in this structure 
![image](https://github.com/SrkPrasadChangala/pneumoniaDetection/assets/77905636/7a807f94-5d4f-4f50-9878-c0c5fe604cd6)


## 3. Data Exploration
Image counts in each class (normal and pneumonia) for both training and test sets are calculated, providing insights into the dataset's distribution and size.
There a total of 5857 x-ray images in this dataset out of which
Images in train dataset
 	Normal Images: 1349 	 Pneumonia Images: 3884
Images in test dataset
 	Normal Images: 234 	 Pneumonia Images: 390 

## 4. Visualization
Random sample images from each class are visualized, offering a qualitative understanding of the data. 
These are the sample visualizations of the x-ray data images used in the project.
![image](https://github.com/SrkPrasadChangala/pneumoniaDetection/assets/77905636/b6be27d6-9e46-4863-b5cc-a0eb97b5e40f)
![image](https://github.com/SrkPrasadChangala/pneumoniaDetection/assets/77905636/f8c6e256-97ea-4b3e-bbca-2305cf0a93ad)

Class distribution is also visualized using a bar chart. It is as shown in the image given
![image](https://github.com/SrkPrasadChangala/pneumoniaDetection/assets/77905636/fb9c8cab-f247-44f2-ad27-3206b3cb6d1a)
As seen the dataset has a higher number of pneumonia images close to 4274 and lesser number of normal images which is close to 1583. Speaking in percentages, the dataset contains 72.9% of Pneumonia images and 27.1% of dataset contains normal images.


## 5. Image Size Analysis
Histograms analyze image size distributions (width and height), and sample images with labels showcase the dataset's diversity. The histograms indicate that image widths and heights in the dataset are normally distributed, peaking around 1250 and 1000 pixels respectively, with few images exceeding 2500 pixels in either dimension. This suggests that standardizing image sizes to a common dimension like 256x256 pixels for model input might be appropriate, balancing detail retention with computational efficiency.
![image](https://github.com/SrkPrasadChangala/pneumoniaDetection/assets/77905636/0370eaf5-459b-4b46-8356-182ca8ca0e27)


## 6. Data Preprocessing
Directory paths, batch size, image size, and validation split are defined for data preprocessing. ImageDataGenerator is utilized for data augmentation and normalization.
During data preprocessing 4187 images of both normal and pneumonia classes are inserted in train_generator,1045 to validation_generator and 624 images to test_generator.

## 7. Model Development (CNN)
A basic CNN model is developed with essential layers. A sequential model in instantiated with a basic convolution layer with a relu activation function and a Max Pooling layer with a pool size of 2X2 and a flatten layer is added to transform the 2D matrix data to a vector and a fully cinnected layer with 128 unite and ReLu activation and finally an output layet with a single unit and a sigmoid classifier for binary classification. The model is compiled with an Adam optimizer and binary crossentropy loss. Training is performed on the training set with validation on a subset.

The results of the basic CNN model on the data are:
```
Model: "cnn_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 254, 254, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 127, 127, 32)      0         
 D)                                                              
                                                                 
 flatten (Flatten)           (None, 516128)            0         
                                                                 
 dense (Dense)               (None, 128)               66064512  
                                                                 
 dense_1 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 66065537 (252.02 MB)
Trainable params: 66065537 (252.02 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/5
66/66 [==============================] - 157s 2s/step - loss: 1.6165 - accuracy: 0.8295 - val_loss: 0.2581 - val_accuracy: 0.8909
Epoch 2/5
66/66 [==============================] - 122s 2s/step - loss: 0.2380 - accuracy: 0.9040 - val_loss: 0.2091 - val_accuracy: 0.9148
Epoch 3/5
66/66 [==============================] - 124s 2s/step - loss: 0.1993 - accuracy: 0.9193 - val_loss: 0.2005 - val_accuracy: 0.9072
Epoch 4/5
66/66 [==============================] - 123s 2s/step - loss: 0.1869 - accuracy: 0.9243 - val_loss: 0.2207 - val_accuracy: 0.9062
Epoch 5/5
66/66 [==============================] - 122s 2s/step - loss: 0.1689 - accuracy: 0.9341 - val_loss: 0.2081 - val_accuracy: 0.9120
```

## 8. Model Improvement (CNN with Data Augmentation)
Data is augmented by rotating images up to 0.2 degrees for enhanced model generalisation.Dropout is added for regularization, and the model is recompiled and trained with early stopping. The results are:
```
Model: "cnn_model_updated"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 sequential_3 (Sequential)   (None, 256, 256, 3)       0         
                                                                 
 conv2d_7 (Conv2D)           (None, 254, 254, 64)      1792      
                                                                 
 max_pooling2d_7 (MaxPoolin  (None, 127, 127, 64)      0         
 g2D)                                                            
                                                                 
 conv2d_8 (Conv2D)           (None, 125, 125, 128)     73856     
                                                                 
 max_pooling2d_8 (MaxPoolin  (None, 62, 62, 128)       0         
 g2D)                                                            
                                                                 
 flatten_4 (Flatten)         (None, 492032)            0         
                                                                 
 dropout_5 (Dropout)         (None, 492032)            0         
                                                                 
 dense_12 (Dense)            (None, 256)               125960448 
                                                                 
 dense_13 (Dense)            (None, 1)                 257       
                                                                 
=================================================================
Total params: 126036353 (480.79 MB)
Trainable params: 126036353 (480.79 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/15
66/66 [==============================] - 136s 2s/step - loss: 1.4160 - accuracy: 0.6883 - val_loss: 0.4488 - val_accuracy: 0.7589
Epoch 2/15
66/66 [==============================] - 128s 2s/step - loss: 0.3852 - accuracy: 0.8292 - val_loss: 0.3918 - val_accuracy: 0.8344
Epoch 3/15
66/66 [==============================] - 127s 2s/step - loss: 0.3147 - accuracy: 0.8648 - val_loss: 0.4119 - val_accuracy: 0.8268
Epoch 4/15
66/66 [==============================] - 127s 2s/step - loss: 0.3260 - accuracy: 0.8586 - val_loss: 0.3512 - val_accuracy: 0.8574
Epoch 5/15
66/66 [==============================] - 127s 2s/step - loss: 0.2966 - accuracy: 0.8772 - val_loss: 0.4081 - val_accuracy: 0.8335
Epoch 6/15
66/66 [==============================] - 124s 2s/step - loss: 0.3073 - accuracy: 0.8682 - val_loss: 0.3828 - val_accuracy: 0.8421
Epoch 7/15
66/66 [==============================] - 125s 2s/step - loss: 0.2763 - accuracy: 0.8827 - val_loss: 0.3783 - val_accuracy: 0.8555
```

## 9. Transfer Learning (ResNet50)
The ResNet50 model, pretrained on ImageNet, is utilized with additional layers for pneumonia detection. The model is compiled, and training is performed with early stopping. The results of ResNet50 with weights from ImageNet are given below:
```
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_6 (InputLayer)        [(None, 256, 256, 3)]     0         
                                                                 
 tf.__operators__.getitem_2  (None, 256, 256, 3)       0         
  (SlicingOpLambda)                                              
                                                                 
 tf.nn.bias_add_2 (TFOpLamb  (None, 256, 256, 3)       0         
 da)                                                             
                                                                 
 resnet50 (Functional)       (None, None, None, 2048   23587712  
                             )                                   
                                                                 
 global_average_pooling2d_2  (None, 2048)              0         
  (GlobalAveragePooling2D)                                       
                                                                 
 dense_14 (Dense)            (None, 128)               262272    
                                                                 
 dropout_6 (Dropout)         (None, 128)               0         
                                                                 
 dense_15 (Dense)            (None, 1)                 129       
                                                                 
=================================================================
Total params: 23850113 (90.98 MB)
Trainable params: 262401 (1.00 MB)
Non-trainable params: 23587712 (89.98 MB)
_________________________________________________________________
Epoch 1/5
66/66 [==============================] - 133s 2s/step - loss: 0.5917 - accuracy: 0.7296 - val_loss: 0.5741 - val_accuracy: 0.7426
Epoch 2/5
66/66 [==============================] - 142s 2s/step - loss: 0.5668 - accuracy: 0.7421 - val_loss: 0.5529 - val_accuracy: 0.7426
Epoch 3/5
66/66 [==============================] - 126s 2s/step - loss: 0.5570 - accuracy: 0.7421 - val_loss: 0.5412 - val_accuracy: 0.7426
Epoch 4/5
66/66 [==============================] - ETA: 0s - loss: 0.5461 - accuracy: 0.7421Restoring model weights from the end of the best epoch: 1.
66/66 [==============================] - 126s 2s/step - loss: 0.5461 - accuracy: 0.7421 - val_loss: 0.5453 - val_accuracy: 0.7426
Epoch 4: early stopping
```
![image](https://github.com/SrkPrasadChangala/pneumoniaDetection/assets/77905636/78ef50eb-e23a-4a7d-8300-ca5cd49a6b83)
## 10. Model Evaluation
The performance of both the CNN and ResNet models is evaluated on the test set, reporting test accuracy for each model. The results of evaluation of two models are given below:
```
10/10 [==============================] - 8s 750ms/step - loss: 0.4257 - accuracy: 0.8061
CNN Model Test Accuracy: 80.61%
10/10 [==============================] - 7s 710ms/step - loss: 0.7402 - accuracy: 0.6250
ResNet Model Test Accuracy: 62.50%
```

## Conclusion
The development process involves crucial steps such as dataset acquisition, exploration, visualization, preprocessing, model development, and evaluation. The use of both a basic CNN and a transfer learning approach (ResNet50) demonstrates the flexibility and adaptability in choosing models based on the project requirements. The final evaluation provides insights into the models' performance on previously unseen data, which is vital for real-world applications.
