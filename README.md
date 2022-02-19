# Medical Imaging Autoencoder ![Python3Ver](https://img.shields.io/badge/Scripting-Python%203-red) ![TFKeras](https://img.shields.io/badge/TensorFlow-Keras-yellow) ![CV](https://img.shields.io/badge/Computer-Vision-green) ![CI](https://img.shields.io/badge/Cancer-Imaging-orange) ![AI](https://img.shields.io/badge/Artificial-Intelligence-lightgrey) ![GenModels](https://img.shields.io/badge/Generative-Models-blue)

## Overview

This program is a retrofitted Autoencoder. It can currently be applied to clinical images such as cancer histopathological slides. The imaging set this model pertains to is the BreakHis Breast cancer dataset found here: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/. This is a high-quality image set with several different levels of magnification available for a given slide. The images are all multi-channel (i.e. RGB) and are in the lossless PNG format. 
This architecturally-specific Autoencoder program can effectively be used to learn ALL of the ultra-fine patterns found within the data set.

(Note) This program requires Python 3.7.

## Autoencoder

An Autoencoder is a neural network-based compression and generative model. It relies on a learning paradigm dubbed self-supervision: The labels enabling us to measure output quality is generated from the input data itself. Images can be treated as complex multi-dimensional distributions; After all, if those were not distributions, an image would simply be static noise. The autoencoder in essence attempts to learn complex patterns that are common and recurring across images of a specific domain (e.g. images of faces).

## BreakHis Image Set

## Successful Autoencoder Architecture

## Early Failed Attempt Example

## Later Successful Examples

## Results with MNIST

We use:
- A sample of 70,000 images of hand-written digits
- Two simple 3-layers encoder and decoder neural networks
- A batch size of 10
- Five training epochs


