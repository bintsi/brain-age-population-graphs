# A Comparative Study of Population-Graph Construction Methods and Graph Neural Networks for Brain Age Regression

## About
This is a Pytorch Lightning implementation for the paper 
[A Comparative Study of Population-Graph Construction Methods and Graph Neural Networks for Brain Age Regression](https://arxiv.org/abs/2309.14816)
(MICCAI 2023) by Kyriaki-Margarita Bintsi, Tamara T. Mueller, Sophie Starck, Vasileios Baltatzis, Alexander Hammers, and Daniel Rueckert

## Dataset
The dataset used for this paper is the UK Biobank. Since the data is not public, we cannot share the csv files.
You need to put the csv files in the data folder that is available.
The format that the csvs need to have is the following:
train.csv, val.csv, test.csv

For every csv:
Column 0: eid
Column 1: label (age)
Column 2-22: Non-imaging phenotypes
Column 22-90: Imaging phenotypes

## Training
To train a model for the hyperparameters chosen for the regression task run the following command:
`python train_static_graph.py`

## Reference
If you find the code useful, pleace cite: 
```
@article{bintsi2023comparative,
  title={A Comparative Study of Population-Graph Construction Methods and Graph Neural Networks for Brain Age Regression},
  author={Bintsi, Kyriaki-Margarita and Mueller, Tamara T and Starck, Sophie and Baltatzis, Vasileios and Hammers, Alexander and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2309.14816},
  year={2023}
}
```