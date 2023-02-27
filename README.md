# COMUS
This is the official PyTorch implementation for the [paper](https://aclanthology.org/2022.acl-long.408/):
> Zheng Gong*, Kun Zhou*, Xin Zhao, Jing Sha, Shijin Wang, Ji-Rong Wen. Continual Pre-training of Language Models for Math Problem Understanding with Syntax-Aware Memory Network. ACL 2022.

## Overview

We propose ***COMUS***, a new approach to **co**ntinually pre-train language models for **m**ath problem **u**nderstanding with **s**yntax-aware memory network. We construct math syntax graph to model the structural semantic information, and then design the syntax-aware memory networks to deeply fuse the features from the graph and text. We finally devise three continual pre-training tasks to further align and fuse the representations of the text and math syntax graph.

![](figure/model.png)

## Requirements

```
# for model
torch==1.10.0
transformers==4.6.0
datasets==1.1.3
dgl==0.8.x
# for data process
jieba
sympy
bs4
timeout_decorator
```

## Dataset

Data cannot be shared temporarily due to commercial reasons. We put the preprocessing code of the data in the `data` folder as a reference.

## Training

```bash
bash scripts/run_pretrain.sh
```

## Citation

Please consider citing our paper if you use our codes.

```bibtex
@inproceedings{gong-etal-2022-continual,
    title = "Continual Pre-training of Language Models for Math Problem Understanding with Syntax-Aware Memory Network",
    author = "Gong, Zheng  and
      Zhou, Kun  and
      Zhao, Xin  and
      Sha, Jing  and
      Wang, Shijin  and
      Wen, Ji-Rong",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.408",
    doi = "10.18653/v1/2022.acl-long.408",
    pages = "5923--5933",
}
```