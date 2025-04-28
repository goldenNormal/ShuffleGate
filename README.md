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



## Citation
If you find this work useful in your research, please consider citing our paper:
```
@misc{shuffle_gate,
      title={ShuffleGate: An Efficient and Self-Polarizing Feature Selection Method for Large-Scale Deep Models in Industry}, 
      author={Yihong Huang and Chen Chu and Fan Zhang and Fei Chen and Yu Lin and Ruiduan Li and Zhihao Li},
      year={2025},
      eprint={2503.09315},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.09315}, 
}
```
