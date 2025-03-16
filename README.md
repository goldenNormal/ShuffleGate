# ShuffleGate: An Efficient and Self-Polarizing Feature Selection Method for Large-Scale Deep Models in Industry

This repository contains the official implementation of ShuffleGate, a novel feature selection method for deep learning models that achieves efficient and self-polarizing feature selection in industrial scenarios.

⭐ If you find this repository helpful, please consider giving it a star! ⭐

Hope you like our public code! Enjoy using ShuffleGate for straightforward feature selection.

## Getting Started

### Dataset
The preprocessed DRS dataset is available on Huggingface:
- Dataset: [DRS-dataset](https://huggingface.co/datasets/yihong-1101/DRS-dataset)
- Download the dataset to the `quick_data` directory

Note: `utils/datasets.py` describes how we transform the original ERASE dataset [ERASE_Dataset](https://huggingface.co/datasets/Jia-py/ERASE_Dataset) to our dataset, in order to achieve a smaller storage and much more efficient read files speed.

### Prerequisites
All required packages are listed in `requirements.txt`.

## Running Experiments

The experimental pipeline consists of two stages:

### 1. Search and Retrain Stage
Run all feature selection methods:
```bash
python search_and_retrain.py
```
Results will be saved in the `exp_save` directory.

### 2. Results Analysis
Analyze the experimental results using:
```bash
jupyter notebook agg_results.ipynb
```

## Project Structure
```
.
├── quick_data/          # Dataset directory
├── exp_save/           # Experimental results
├── search_and_retrain.py    # Main experiment script
├── agg_results.ipynb   # Results analysis notebook
└── requirements.txt    # Package dependencies
```




## Citation
If you find this work useful in your research, please consider citing our paper:
```
[Citation will be added upon publication]
```
