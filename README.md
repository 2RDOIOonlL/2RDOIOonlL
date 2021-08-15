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

Secondly, we provide results on CIFAR-10-C using the following corruption types: `brightness, defocus blur, motion blur, gaussian noise, pixelate`. (Metric used: Softmax Entropy, Ensemble PE and DDU density)

<span>
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_brightness.png" width="39%">
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_defocus_blur.png" width="39%">
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_motion_blur.png" width="39%">
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_gaussian_noise.png" width="39%">
<img src="https://github.com/2RDOIOonlL/2RDOIOonlL/blob/main/cifar10_c_pixelate.png" width="39%">
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


Are there any additional experiments you could suggest to study the properties of the uncertainty? We will gladly try to provide additional results during the rebuttal phase.
