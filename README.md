# Learnable Pillar-based Re-ranking for Image-Text Retrieval (LeaPRR)

PyTorch code of the SIGIR'23 paper "Learnable Pillar-based Re-ranking for Image-Text Retrieval". 

## Introduction

Image-text retrieval aims to bridge the modality gap and retrieve cross-modal content based on semantic similarities. Prior work usually focuses on the pairwise relations (i.e., whether a data sample matches another) but ignores the higher-order neighbor relations (i.e., a matching structure among multiple data samples). Re-ranking, a popular post-processing practice, has revealed the superiority of capturing neighbor relations in single-modality retrieval tasks. However, it is ineffective to directly extend existing re-ranking algorithms to image-text retrieval. In this paper, we analyze the reason from four perspectives, i.e., generalization, flexibility, sparsity, and asymmetry, and propose a novel learnable pillar-based re-ranking
paradigm. Concretely, we first select top-ranked intra- and intermodal neighbors as pillars, and then reconstruct data samples with the neighbor relations between them and the pillars. In this way,
each sample can be mapped into a multimodal pillar space only using similarities, ensuring generalization. After that, we design a neighbor-aware graph reasoning module to flexibly exploit the relations
and excavate the sparse positive items within a neighborhood. We also present a structure alignment constraint to promote cross-modal collaboration and align the asymmetric modalities. On top of
various base backbones, we carry out extensive experiments on two benchmark datasets, i.e., Flickr30K and MS-COCO, demonstrating the effectiveness, superiority, generalization, and transferability of
our proposed re-ranking paradigm.

![model](/fig/model.png)

## Requirements 

We recommended the following dependencies.

- Python 3.7 
- [PyTorch](http://pytorch.org/) (1.7.0)
- [NumPy](http://www.numpy.org/) (1.18.0)
- [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Download data

We use Flickr30k and MS-COCO datasets and splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from their original sources [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

The base models used in this paper include [SCAN](https://github.com/kuanghuei/SCAN), [VSRN](https://github.com/KunpengLi1994/VSRN), [CAMERA](https://github.com/LgQu/CAMERA), [VSE_infinity](https://github.com/woodfrog/vse_infty), and [DIME](https://github.com/LgQu/DIME). We first evaluate the base models and calculate initial ranking results and similarities using the provided checkpoints in these repositories. After that, we preprocess the initial ranking results and similarities in a batch fashion and save the structured data for model training. All the structured similarities and initial ranking results needed for reproducing the experiments in the paper, can be downloaded [here](https://drive.google.com/drive/folders/1VMd0hVw8-0s41ljNQM_Sic0RbiBU8awN?usp=sharing). 

We refer to the path of extracted files as `$DATA_PATH`. 

## Train new models

Take training the model based on DIME on the Flickr30k dataset for an example: 

```bash
python main.py --logger_path $LOGGER_PATH --data_path $DATA_PATH --dataset flickr --base_model DIME_ensemble
```

By setting `dataset` and `base_model`, one can train models on other datasets and base models. 



## Evaluate pre-trained models

Download pretrained models [here](https://drive.google.com/drive/folders/1VX1GYzuMLwfgJa5x1WDtKrrG5DMaduFW?usp=sharing) and put them in `LOGGER_PATH`. Run main.py by setting `only_test`:

```bash
python main.py --logger_path $LOGGER_PATH --data_path $DATA_PATH --dataset flickr --base_model DIME_ensemble --only_test checkpoint_best
```



## Reference

```
@inproceedings{qu2023learnable,
  title={Learnable Pillar-based Re-ranking for Image-Text Retrieval},
  author={Qu, Leigang and Liu, Meng and Wang, Wenjie and Zheng, Zhedong and Nie, Liqiang and Chua, Tat-Seng},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1252--1261},
  year={2023}
}
```