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


## Ablation: VGG with SN

In this section, we present results for VGG models trained using spectral normalization (SN). The first table below is for VGG models trained on CIFAR-10 and the next table is for models trained on CIFAR-100. We note that SN does not improve the OoD detection performance for VGG models. A possible explanation for this observation is that the VGG architecture, even with spectral normalisation, does not encourage sensitivity to changes in the input space, unlike residual architectures. This can be deduced from Tables 3 and 4 in the appendix, and from Figure 1(d) in the paper. From the theory perspective, in SNGP the residual connections are required in the proof which shows that the bounded spectral norm of the weights affects the function's Lipschitz constant.


| SN  |       Method       |  Accuracy   |    ECE     | AUROC SVHN  | AUROC CIFAR100 |
|-----|--------------------|-------------|------------|-------------|----------------|
| No  | Softmax            | 93.63+-0.04 | 1.64+-0.03 | 85.76+-0.84 | 82.48+-0.14    |
| Yes | Softmax            | 93.56+-0.03 | 1.69+-0.04 | 86.55+-0.51 | 82.40+-0.09    |
| No  | Energy-based model | 93.63+-0.04 | 1.64+-0.03 | 84.24+-1.04 | 81.91+-0.17    |
| Yes | Energy-based model | 93.56+-0.03 | 1.69+-0.04 | 84.77+-0.68 | 81.79+-0.11    |
| No  | GMM Density        | 93.63+-0.04 | 1.64+-0.03 | 89.25+-0.36 | 86.55+-0.10    |
| Yes | GMM Density        | 93.56+-0.03 | 1.69+-0.04 | 89.51+-0.33 | 86.52+-0.12    |

-- with CIFAR-10 as iD dataset


| SN  |       Method       |  Accuracy   |    ECE     | AUROC SVHN  |
|-----|--------------------|-------------|------------|-------------|
| No  | Softmax            | 73.48+-0.05 | 4.46+-0.05 | 76.73+-0.72 |
| Yes | Softmax            | 73.56+-0.05 | 4.49+-0.06 | 76.43+-0.74 |
| No  | Energy-based model | 73.48+-0.05 | 4.46+-0.05 | 77.70+-0.86 |
| Yes | Energy-based model | 73.56+-0.05 | 4.49+-0.06 | 77.07+-0.84 |
| No  | GMM Density        | 73.48+-0.05 | 4.46+-0.05 | 75.65+-0.95 |
| Yes | GMM Density        | 73.56+-0.05 | 4.49+-0.06 | 75.05+-1.41 |

-- with CIFAR-100 as iD dataset

## Ablation: Ensemble with SN

In this section, we present results for ensembles of models trained using spectral normalization (SN). The first table below is for models trained on CIFAR-10 and the next table is for models trained on CIFAR-100. All the results have been reported using predictive entropy as the uncertainty metric. 

We note that in general, ensembles of models trained with SN don't outperform those trained without SN. Note that although the AUROC for the WRN+SN ensemble is slightly higher than the WRN ensemble, it is still within the standard error and hence, not statistically significant.

|     Model       |  Accuracy   |    ECE     | AUROC SVHN  | AUROC CIFAR-100 |
|-----------------|-------------|------------|-------------|-----------------|
| WRN Ensemble    | 96.59+-0.02 | 0.76+-0.03 | 97.73+-0.31 | 92.13+-0.02     |
| WRN+SN Ensemble | 96.68+-0.03 | 0.82+-0.05 | 97.59+-0.08 | 91.30+-0.07     |
| VGG Ensemble    | 94.9+-0.05  | 2.03+-0.03 | 92.80+-0.18 | 89.01+-0.08     |
| VGG+SN Ensemble | 94.96+-0.05 | 2.10+-0.08 | 90.36+-0.23 | 88.25+-0.10     |

-- with CIFAR-10 as iD dataset

|     Model       |  Accuracy   |     ECE     | AUROC SVHN  |
|-----------------|-------------|-------------|-------------|
| WRN Ensemble    | 82.79+-0.10 | 3.32+-0.09  | 79.54+-0.91 |
| WRN+SN Ensemble | 83.06+-0.07 | 2.17+-0.1   | 80.30+-0.85 |
| VGG Ensemble    | 77.84+-0.11 | 5.32+-0.10  | 79.62+-0.73 |
| VGG+SN Ensemble | 77.98+-0.06 | 3.001+-0.05 | 73.64+-1.03 |

-- with CIFAR-100 as iD dataset

## Toy Example: Different Objectives Have Different Optima (Proposition 4.3)

In the following toy example, we retrace proposition 4.3 and visualize that different objectives lead to different optima. We exactly follow the construction in the proof for 4.3 in the appendix. We hope this provides an additional more visual explanation and examination of the proposition.

### (Class) Density Plots
<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/separate_objective_1.png" width="80%">
</span>

### Entropy Plots

<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/separate_objective_2.png" width="80%">
</span>

-- Yellow star at (0, -5) marked for comparison of the entropy predictions when trained on different objectives.

### Objective Scores


| Objective | `H_θ[Y\|Z]` | `H_θ[Y,Z]` | `H_θ[Z]` |
|--|--|--|--|
| `min H_θ[Y\|Z]`         | **0.1794** | 5.4924   | 5.2995   |
| `min H_θ[Y,Z]`         | 0.2165   | **4.9744** | 4.7580   |
| `min H_θ[Z]`           |  n/a     |  n/a     | **4.7073** |
 

### Description

To explain Proposition 4.3 in an intuitive way, we focus on on a simple 2D toy case and fit a GMM using the different objectives. We sample "latents" z from 3 Gaussians (each representing a different class y) with 4% label noise. (As mentioned in the proof in the appendix, this is an easy way to see how the objectives will lead to different optima.)

**`min H_θ[Y|Z].`** A softmax linear layer is equivalent to an LDA (Linear Discriminant Analysis) with conditional likelihood as detailed in "Machine Learning: A Probabilistic Perspective" by Murphy, for example. We optimize an LDA with the usual objective "`min -1/N \sum \log p(y|z)`", i.e. the cross-entropy of `p(y|z)` or  (average) negative log-likelihood NLL. Following the appendix, we use the short-hand "`min H_θ[Y|Z]`" for this cross-entropy.

Because we optimize only `p(y|z)`, `p(z)` does not affect the objective and is thus not optimized. Indeed, the components do not actually cover the latents well, as can be seen in the first density plot. However, it does provide the lowest NLL.

**`min H_θ[Y,Z].`** We optimize a GDA for the combined objective "`min -1/N \sum \log q(y, z)`", i.e. the cross-entropy of `q(y, z)`. We use the short-hand "`min H_θ[Y|Z]`" for this.

**`min H_θ[Z].`** We optimize a GMM for the objective "`min -1/N \sum \log q(z)`", i.e. the cross-entropy of `q(z)`. We use the short-hand "`min H_θ[Z]`" for this.

We see that each solution minimizes its objective the best. The GMM provides the best density model (best fit according to the entropy), while the LDA (like a softmax linear layer) provides the best NLL for the labels. The GDA provides an almost as good density model.

**Entropy.** Looking at the entropy plots, we first notice that the LDA solution optimized for `min H_θ[Y|Z]` has a wide decision boundary. This is due to the overlap of the Gaussian components, which is necessary to provide the right aleatoric uncertainty. 

Optimizing the negative log-likehood `-\log p(y|z)` is a proper scoring rule, and hence is optimized for calibrated predictions. 

Compared to this, the GDA solution (optimized for `min H_θ[Y, Z]`) has a much narrower decision boundary and cannot capture aleatoric uncertainty as well. This is reflected in the higher NLL. Moreover, unlike for LDA, GDA decision boundaries behave differently than one would naively expect due to the untied covariance matrices. They can be curved and the decisions change far away from the data. (See also "Machine Learning: A Probabilistic Perspective" by Murphy.)

To show the difference between the two objectives we have marked one point *(0, -5)* with a yellow star. Under the first objective `min H_θ[Y,Z]`, it has high aleatoric uncertainty (high entropy), as seen in the left entropy plot, while under the second objective (`min H_θ[Y,Z]`), it is only assigned very low entropy. The GDA optimized for the second objective thus is overconfident.

We do show an entropy plot for `Y|Z` for the third objective `min H_θ[Z]` as it does not try to learn class assignments, and hence the different components do not map to classes in order.

We hope this provides good intuitions for the statement of Proposition 4.3. Intuitively, for aleatoric uncertainty, the Gaussian components need to overlap to express high aleatoric uncertainty (uncertain labelling). At the same time, this necessarily provides looser density estimates. On the other hand, the GDA density is much tighther, but this comes at the cost of NLL for classification because it cannot express aleatoric uncertainty that well. This visualizes that the objectives trade-off between each other, and why we use the softmax layer trained for `p(y|z)` for classification and aleatoric uncertainty, and GDA as density model for `q(z)`.

## Algorithm Box

In order to shorten the algorithm section, we plan on merging the text in Section 5 (Algorithm) of the paper with Appendix B (Additional Architectural Changes). We will rename Appendix B to "DDU Algorithm". In the main paper, we instead plan on putting a succinct algorithm box as shown below.

<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/ddu_algorithm.PNG" width="60%">
</span>

For the ease of readers, we also plan to add in a pseudocode of the algorithm, as shown below, in Appendix B.
<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/ddu_pseudocode.PNG" width="60%">
</span>

Are there any additional experiments you could suggest to study the properties of the uncertainty? We will gladly try to provide additional results during the rebuttal phase.
