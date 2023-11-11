# Transfer-learning-with-CNN-for-EEG-signals-in-diagnosing-Schizophrenia-patients..


We propose an automated method that utilizes transfer learning with deep convolutional neural networks (CNNs) to differentiate individuals with SZ from healthy controls. Initially, we transform EEG signals into images using a time-frequency technique known as the continuous wavelet transform (CWT). Subsequently, these EEG signal images are inputted into four widely used pre-trained CNN models: AlexNet, ResNet-18, VGG-19, and Inception-v3. The convolutional and pooling layers of these models produce deep features, which are then utilized as input for a support vector machine (SVM) classifier.
