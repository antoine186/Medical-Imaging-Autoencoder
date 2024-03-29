# Medical Imaging Autoencoder ![Python3Ver](https://img.shields.io/badge/Scripting-Python%203-red) ![TFKeras](https://img.shields.io/badge/TensorFlow-Keras-yellow) ![CV](https://img.shields.io/badge/Computer-Vision-green) ![CI](https://img.shields.io/badge/Cancer-Imaging-orange) ![AI](https://img.shields.io/badge/Artificial-Intelligence-lightgrey) ![GenModels](https://img.shields.io/badge/Generative-Models-blue)

## Overview

This program is a retrofitted Autoencoder. It can currently be applied to clinical images such as cancer histopathological slides. The imaging set this model pertains to is the BreakHis Breast cancer dataset found here: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/. This is a high-quality image set with several different levels of magnification available for a given slide. The images are all multi-channel (i.e. RGB) and are in the lossless PNG format. 
This architecturally-specific Autoencoder program can effectively be used to learn ALL of the ultra-fine patterns found within the data set.

(Note) This program requires Python 3.7.

## Autoencoder

An Autoencoder is a neural network-based compression and generative model. It relies on a learning paradigm dubbed self-supervision: The labels enabling us to measure output quality is generated from the input data itself. Images can be treated as complex multi-dimensional distributions; After all, if those were not distributions, an image would simply be static noise. The autoencoder in essence attempts to learn complex patterns that are common and recurring across images of a specific domain (e.g. images of faces).

## BreakHis Image Set

The image set is a high-quality one drawn from 82 patients using different microscopy magnification factors (40X, 100X, 200X, and 400X). The size of the dataset is unusually large with a total of 7,909 exemplars. 
The data is inherently hierarchical as can be seen in the figure below. At a first top level, we have the global distinction between benign and malignant tissue. The benign set contains 2,480 exemplars and the malignant set contains 5,429 exemplars. This would normally constitute a slight degree of class imbalance, however the sample size is so large that such effect should be entirely smoothed out.
Further down and at a lower second level, the data can be subdivided into 4 subclasses each in order to represent all the possible subclasses of both benign and malignant biological manifestations.

<p align="center">
  <img src="https://github.com/antoine186/Medical-Imaging-Autoencoder/blob/master/blob/BrChart.png" alt="alt text" width=60% height=60%>
</p>

## Successful Autoencoder Architecture

This Autoencoder is in essence a Deep Neural Network model. Let us first explain the purpose of its deepness: It needs to be able to capture the multi-dimentional and highly varied (i.e. if one inspects the image set, they will find a great number of subclasses with many benign exemplars resembling strongly malignant exemplars to the naked eye!!!) patterns of cancer/healthy features within the biopsies. The ultimate goal of the Autoencoder is to intervene within the training of a GAN model - outside the scope of this repo - for the synthetic generation of cancer images. The GAN/Autoencoder experiment setup can be found in the image below:

<p align="center">
  <img src="https://github.com/antoine186/Medical-Imaging-Autoencoder/blob/master/blob/CompressionAIDesign.png" alt="alt text" width=60% height=60%>
</p>

The Autoencoder helps to compress and speed up the learning process of the GAN, which is known to have incredibly fickle training behaviour and to require inordinate and frequently impractical amounts of computing power.

Our Deep Autoencoder is comprised of two parts: the encoder deep network and the decoder deep network. The encoder model is made up of 3 convolutional layers interleaved with 2 max-pooling layers. The latter layers perform heuristic and empirically proven downsampling of an image using pixel value thresholding. It also acts to greatly limit the complexity of an effective model without much impacting performance. Subsequently, the decoder network is made up of 3 deconvolutional layers (i.e. combining convolutional layers with up-sampling layers) and a final convolutional layer. The last layer is counter-intuitive, but required in order to obtain the latent space exemplar dimension we aimed for, which was a 3-dimensional cube with 58x88x16 dimensions.

## Early Failed Attempt Example

Our first half-dozen attempts were failures. Even though an Autoencoder is much better behaved and easier to train than the much more complex GAN model, training an autoencoder is by no means simple in absolute terms. Each failed training attempts' duration ranged between 8 to 16 hours! We essentially encountered severe issues of mode collapse found in the example images below:

<p align="center">
  <img src="https://github.com/antoine186/Medical-Imaging-Autoencoder/blob/master/blob/SmallLossVeryLossy.png" alt="alt text" width=60% height=60%>
</p>

In this first image, the mode collapsed entirely upon complete noise (i.e. it didn't even collapse on a single clearly legible pattern).

<p align="center">
  <img src="https://github.com/antoine186/Medical-Imaging-Autoencoder/blob/master/blob/SmallerLoss.png" alt="alt text" width=60% height=60%>
</p>

In this second image, there is much improvement and the mode collapses more softly, but it is unable to produce any output outside of this likeness range. Additionally, the middle section shows that it also collapses softly on noise.

## Later Successful Examples

In order to train the Autoencoder successfully to the point that it was able to reconstruct the input flawlessly (i.e. images below), we had to make the latent space relatively larger. Going increasingly small in this respect causes the model performance to tend towards a near incomprehensible reconstruction. 
We also decided to use for both the encoder and the decoder network the ReLU activation function. 
We did in fact attempt to use other well-known activation techniques such as the elu, exponential, and selu functions. Those were in the end discarded as they either led to training error rates that were relatively-speaking much higher or to reconstructions of very poor quality.
<br />
&nbsp;  
Perfect reconstruction of a benign exemplar:
<p align="center">
  <img src="https://github.com/antoine186/Medical-Imaging-Autoencoder/blob/master/blob/BenignPerfRecon.png" alt="alt text" width=60% height=60%>
</p>
<br />
Perfect reconstruction of a malignant exemplar:
&nbsp;  
<p align="center">
  <img src="https://github.com/antoine186/Medical-Imaging-Autoencoder/blob/master/blob/MalignantPerfRecon.png" alt="alt text" width=60% height=60%>
</p>

The final successful training params. were as follows:

* 2000 training epochs
* Batch size: 10 <- A higher batch size causes my computer to run out of memory ^^
* 80-20% train-test split
* Loss function: Binary cross-entropy
* Learning algorithm: Adadelta

The training time was upwards of 11 hours 
