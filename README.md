# Additional experiment results

We are pleased to provide additional results that reviewers asked for:

> Dhut: “The experiments are only based on two datasets. More datasets should be included to compare the performance of the proposed method with other existing methods.”

> LLrb: “I would also like to see how to proposed method compares to baselines when trained on MNIST/CIFAR10 and evaluated on increasingly rotated/corrupted versions of the test set.”

> 6uob: “In particular, I want to see more results on challenging OOD detection tasks like CIFAR-corruptions or adversarial examples, which are more practical and meaningful in real-world settings.”

## CIFAR-10 vs Tiny-ImageNet

First, we provide results on CIFAR-10 (iD) vs Tiny-ImageNet (OoD):

AUROC on Tiny-ImageNet for models trained on CIFAR-10:

|          | AUROC (OOD) ⇈  |
|----------|---------------|
| Softmax  | 88.42 +- 0.05 |
| SNGP     | 89.96 +- 0.08 |
| Ensemble | 90.05 +- 0.03 |
| **DDU**  | **90.27 +- 0.07** |

-- AUROC on Tiny-ImageNet for models trained on CIFAR-10
(Measures of uncertainty used: Softmax Entropy, Ensemble PE and DDU density)

Here DDU also outperforms Deep Ensembles as well as the other single-forward-pass models.

## CIFAR-10-C (Corruptions)

Secondly, we provide results on CIFAR-10-C. (Metric used: Softmax Entropy, Ensemble PE and DDU density)

Below, we provide a plot averaged over all corruptions as well as plots for each individual corruption type.

<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c.png" width="49%">
<details>
  <summary><strong>Please expand here to see all individual CIFAR-10-C results</strong></summary>
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_brightness.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_contrast.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_defocus_blur.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_elastic_transform.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_fog.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_frost.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_gaussian_blur.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_gaussian_noise.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_glass_blur.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_impulse_noise.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_jpeg_compression.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_motion_blur.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_pixelate.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_saturate.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_shot_noise.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_snow.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_spatter.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_speckle_noise.png" width="39%">
  <img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_zoom_blur.png" width="39%">
</details>
</span>

Overall, we hope you can agree that DDU performs well. However, we want to stress that the experiments we have run in the paper (CIFAR-10 vs SVHN and vs CIFAR-100 etc.) are generally considered the hard dataset pairs in the field.

## Feature-Space Density and Epistemic Uncertainty

Lastly, we have also run additional experiments to validate the connection between feature-space density and epistemic uncertainty:

We train on increasingly large subsets of DirtyMNIST and evaluate the epistemic and aleatoric uncertainty on DirtyMNIST’s test set using DDU (“GMM density” and Softmax entropy”, respectively).

We see that with larger training sets, the epistemic uncertainty decreases, that is the average feature-space density increases, while the aleatoric uncertainty stays roughly the same.
All of this is consistent with the experiments comparing epistemic and aleatoric uncertainty on increasing training set sizes in “What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?”, table 3.

<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/gmm_density.png" width="39%">
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/softmax_entropy.png" width="39%">
</span>

|  Train Set  | Avg Softmax Entropy (Test Set) ≈ | Avg Log GMM Density (Test Set) ⇈ |
|-----------------|-------------------------|-------------------------|
| 1% of D-MNIST   |                  0.7407 | -2.7268e+14             |
| 2% of D-MNIST   |                  0.6580 | -7.8633e+13             |
| 10% of D-MNIST  |                  0.8295 | -1279.1753              |


## Ablation: Ensemble with SN

In this section, we present results for ensembles of models trained using spectral normalization (SN). The first table below is for models trained on CIFAR-10 and the next table is for models trained on CIFAR-100. We note that in general, ensembles of models trained with SN don't outperform those trained without SN.

|     Model       |  Accuracy   |    ECE     | AUROC SVHN  | AUROC CIFAR-100 |
|-----------------|-------------|------------|-------------|-----------------|
| WRN Ensemble    | 96.59+-0.02 | 0.76+-0.03 | 97.73+-0.31 | 92.13+-0.02     |
| WRN+SN Ensemble | 96.68+-0.03 | 0.82+-0.05 | 97.59+-0.08 | 91.30+-0.07     |
| VGG Ensemble    | 94.9+-0.05  | 2.03+-0.03 | 92.80+-0.18 | 89.01+-0.08     |
| VGG+SN Ensemble | 94.96+-0.05 | 2.10+-0.08 | 90.36+-0.23 | 88.25+-0.10     |


|     Model       |  Accuracy   |     ECE     | AUROC SVHN  |
|-----------------|-------------|-------------|-------------|
| WRN Ensemble    | 82.79+-0.10 | 3.32+-0.09  | 79.54+-0.91 |
| WRN+SN Ensemble | 83.06+-0.07 | 2.17+-0.1   | 80.30+-0.85 |
| VGG Ensemble    | 77.84+-0.11 | 5.32+-0.10  | 79.62+-0.73 |
| VGG+SN Ensemble | 77.98+-0.06 | 3.001+-0.05 | 73.64+-1.03 |


## Ablation: VGG with SN

In this section, we present results for VGG models trained using spectral normalization (SN). The first table below is for VGG models trained on CIFAR-10 and the next table is for models trained on CIFAR-100. We note that SN does not improve the OoD detection performance for VGG models. The VGG architecture, due to the lack of residual connections, does not encourage sensitivity in the feature space. Addition of SN does not help with sensitivity either as SN encourages smoothness in the feature space by upper bounding the Lipschitzness of the model. Hence, we don't see an improvement in performance for VGG models trained with spectral normalization.

| SN  |       Method       |  Accuracy   |    ECE     | AUROC SVHN  | AUROC CIFAR100 |
|-----|--------------------|-------------|------------|-------------|----------------|
| No  | Softmax            | 93.63+-0.04 | 1.64+-0.03 | 85.76+-0.84 | 82.48+-0.14    |
| No  | Energy-based model | 93.63+-0.04 | 1.64+-0.03 | 84.24+-1.04 | 81.91+-0.17    |
| No  | GMM Density        | 93.63+-0.04 | 1.64+-0.03 | 89.25+-0.36 | 86.55+-0.10    |
| Yes | Softmax            | 93.56+-0.03 | 1.69+-0.04 | 86.55+-0.51 | 82.40+-0.09    |
| Yes | Energy-based model | 93.56+-0.03 | 1.69+-0.04 | 84.77+-0.68 | 81.79+-0.11    |
| Yes | GMM Density        | 93.56+-0.03 | 1.69+-0.04 | 89.51+-0.33 | 86.52+-0.12    |


| SN  |       Method       |  Accuracy   |    ECE     | AUROC SVHN  |
|-----|--------------------|-------------|------------|-------------|
| No  | Softmax            | 73.48+-0.05 | 4.46+-0.05 | 76.73+-0.72 |
| No  | Energy-based model | 73.48+-0.05 | 4.46+-0.05 | 77.70+-0.86 |
| No  | GMM Density        | 73.48+-0.05 | 4.46+-0.05 | 75.65+-0.95 |
| Yes | Softmax            | 73.56+-0.05 | 4.49+-0.06 | 76.43+-0.74 |
| Yes | Energy-based model | 73.56+-0.05 | 4.49+-0.06 | 77.07+-0.84 |
| Yes | GMM Density        | 73.56+-0.05 | 4.49+-0.06 | 75.05+-1.41 |

## Algorithm Box

In order to shorten the algorithm section, we plan on merging the text in Section 5 (Algorithm) of the paper with Appendix B (Additional Architectural Changes). We will rename Appendix B to "Explanation of DDU Algorithm". In the main paper, we instead plan on putting a succinct algorithm box as shown below.
<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/ddu_algorithm.PNG" width="60%">
</span>

For the ease of readers, we also plan to add in a pseudocode of the algorithm, as shown below, in Appendix B.
<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/ddu_pseudocode.PNG" width="60%">
</span>

Are there any additional experiments you could suggest to study the properties of the uncertainty? We will gladly try to provide additional results during the rebuttal phase.
