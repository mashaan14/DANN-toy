# DANN PyTorch implementation with 2D toy example
Domain-Adversarial Neural Network (DANN) is one of the well-known benchmarks for domain adaptation tasks. DANN was introduced by these papers:

```
@misc{https://doi.org/10.48550/arxiv.1409.7495,
  url = {https://arxiv.org/abs/1409.7495},
  author = {Ganin, Yaroslav and Lempitsky, Victor},
  title = {Unsupervised Domain Adaptation by Backpropagation},
  publisher = {arXiv},
  year = {2014},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```
@article{https://doi.org/10.48550/arxiv.1505.07818,
  url = {https://arxiv.org/abs/1505.07818},
  author = {Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, François and Marchand, Mario and Lempitsky, Victor},  
  title = {Domain-Adversarial Training of Neural Networks},
  publisher = {arXiv},  
  year = {2015},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

This DANN implementation uses a 2D toy dataset with built-in plots that help to visualize how the DANN algorithm is learning the new features.

## 2D dataset
The code starts by retrieving `source dataset` from data folder. Then it performs a rotation (domain shift) on a copy of the dataset. The rotated dataset is the `target dataset`. Here is a visualization of source and target datasets:
<p align="center">
  <img width="1200" src=dataset.png>
</p>

## Source domain classifier
The function `core.train_src` trains the `feature_extractor` to separate `source class 0` and `source class 1`. Then, the learned model is tested on the test data:

```
Avg Loss = 0.20282, Avg Accuracy = 88.500000%, ARI = 0.59085
```

<p align="center">
  <img width="1200" src="Testing source data using source feature extractor.png">
</p>

Now, we used the same `feature_extractor` to classify `target` samples. Note that we still did not perform domain adaptation:

```
Avg Loss = 0.61630, Avg Accuracy = 81.000000%, ARI = 0.38154
```

<p align="center">
  <img width="1200" src="Testing target data using source feature extractor.png">
</p>

## Domain adaptation
The domain adaptation takes place in `core.train_tgt` function. The goal is to train the `feature_extractor` to learn features for both `source` and `target` smaples. The `feature_extractor` attempts to minimize a loss computed only on `source` samples, since `target` samples do not have labels. The `feature_extractor` is optimized simultaneously with the `discriminator`, which tries to "discriminate" if the sample is coming from `source` or `target` domains. Eventually, the `feature_extractor` will learn features that make the `discriminator` unable to tell which domain the sample is coming from. Now, we can use the `feature_extractor` to classify `target` samples:


```
Avg Loss = 0.26856, Avg Accuracy = 88.000000%, ARI = 0.57547
```
<p align="center">
  <img width="1200" src="Testing target data using target feature extractor.png">
</p>


## Code acknowledgement
I reused some code from this [repository](https://github.com/corenel/pytorch-ADDA).
