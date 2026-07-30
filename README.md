# Image Classification using Convolutional Neural Networks (CNN)

**Author:** Harsh Raj
**Registration Number:** 23MIM10089
**Application Number:** IN26011390
**Batch:** 1A
**Email:** harsh.23mim10089@vitbhopal.ac.in

## Overview

This project builds a Convolutional Neural Network (CNN) using Keras/TensorFlow to classify images of cats and dogs. It covers dataset acquisition, preprocessing, model training, and evaluation.

## Dataset

[Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) via the Kaggle API — ~25,000 labeled images across two classes (Cat, Dog).

## Project Structure

- `Assignment-9-HarshRaj.ipynb` — main notebook: dataset download, preprocessing, CNN architecture, training, and evaluation.

## Pipeline

1. **Dataset Acquisition** — downloads the dataset via the Kaggle API.
2. **Exploration** — inspects class counts, image dimensions, and sample images.
3. **Preprocessing** — filters corrupted files, splits 80/20 (stratified), and builds `ImageDataGenerator` pipelines.
4. **Model** — 3 convolution + max-pooling blocks, followed by a dense layer with dropout and a sigmoid output.
5. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, and accuracy/loss curves.

## Requirements

```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
kaggle
```

## Setup & Usage

1. Install dependencies: `pip install -r requirements.txt` (or install the packages above individually).
2. Get a Kaggle API key from your [Kaggle account settings](https://www.kaggle.com/settings) (Create New Token).
3. Open `Assignment-9-HarshRaj.ipynb` and run all cells — you'll be prompted for your Kaggle username and API key to download the dataset.

## Notes

- Model outputs (accuracy, plots) are generated fresh each run and depend on the training environment (GPU/CPU, TensorFlow version, random seed).
- Adjust `epochs`, batch size, or add data augmentation if you see overfitting (training accuracy much higher than validation accuracy).
