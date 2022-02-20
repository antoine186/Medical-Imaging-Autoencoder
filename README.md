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

## Early Failed Attempt Example

## Later Successful Examples

## Results with MNIST

We use:
- A sample of 70,000 images of hand-written digits
- Two simple 3-layers encoder and decoder neural networks
- A batch size of 10
- Five training epochs


