# AutoEncoder ![javascript](https://img.shields.io/badge/-javascript-red) ![HTML](https://img.shields.io/badge/-HTML-orange) ![CSS](https://img.shields.io/badge/-css-blue) ![Node.JS](https://img.shields.io/badge/Node.JS-12.18.3-brightgreen) ![npm](https://img.shields.io/badge/npm-live--server-yellow)

## Overview

This program is a retrofitted Variational Autoencoder. It can currently be applied to clinical images such as cancer histopathological slides.

(Note) This program requires Python 3.7.

## VAE

A Variational Autoencoder is a neural network-based compression and generative model. It relies on a learning paradigm dubbed self-supervision: The labels enabling us to measure output quality is generated from the input data itself. Images can be treated as complex multi-dimensional distributions; After all, if those were not distributions, an image would simply be static noise. VAE in essence attempts to learn complex patterns that are common and recurring across images of a specific domain (e.g. images of faces).

## Results with MNIST

We use:
- A sample of 70,000 images of hand-written digits
- Two simple 3-layers encoder and decoder neural networks
- A batch size of 10
- Five training epochs


