# Adversarial-Robustness-and-Probabilistic-Calibration

This project investigates the trade-off between **adversarial robustness** and **probabilistic calibration** in deep neural networks. Using the **CIFAR-10** dataset, we implement and compare four configurations across a **Small CNN** and a **ResNet-18** under standard and FGSM-based adversarial training, evaluating robustness against FGSM and PGD attacks and measuring calibration via ECE, NLL, and reliability diagrams.

The project was developed for the **Neural Networks** course at Universidad Carlos III de Madrid (UC3M).

---

## Project Objective

The main goal is to explore the **Robustness vs. Calibration trade-off** through a structured set of experiments:

1. **Standard Training (Baseline):** Train a Small CNN on CIFAR-10 and measure baseline accuracy and calibration.
2. **Attacking the Model:** Attack the baseline with single-step (FGSM) and iterative (PGD) attacks to demonstrate its vulnerability.
3. **Adversarial Training (Defense):** Retrain the Small CNN injecting FGSM examples during the training loop.
4. **Trade-off Analysis:** Evaluate the defended model — does it survive PGD? Does confidence degrade gracefully as ε increases?
5. **Scaling Up (ResNet-18):** Repeat all experiments with ResNet-18 to assess whether increased capacity helps balance robustness and calibration.

---

## Methodology

The project is implemented in Python using **PyTorch** and follows a structured experimental pipeline:

- **Dataset:** CIFAR-10 — 50,000 training and 10,000 test images across 10 classes (32×32 RGB). Loaded via `torchvision.datasets.CIFAR10` with `download=True`, no data augmentation to avoid contaminating calibration measurements.
- **Models:**
  - *Small CNN:* Custom 4-layer convolutional network with ReLU activations and 2×2 max-pooling, reducing spatial resolution from 32×32 to 2×2. Two fully-connected layers produce class logits. ~2.4M parameters.
  - *ResNet-18:* Standard ResNet-18 adapted for CIFAR-10 — initial 7×7 convolution replaced with 3×3 stride-1, early max-pooling removed. ~11.2M parameters.
- **Training:** Adam optimizer, cross-entropy loss, 15 epochs. Standard training at lr=1e-3; adversarial training at lr=1e-3 (Small CNN) and lr=1e-4 (ResNet-18).
- **Attacks:**
  - *FGSM:* Single-step attack with perturbation ε.
  - *PGD-10:* Iterative attack over 10 steps with step size α=ε/4 and random initialisation within the ε-ball. Evaluation sweep: ε ∈ {0, 4/255, 8/255, 12/255, 16/255}.
- **Calibration Metrics:** Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), and reliability diagrams.

---

## Results Summary

| Architecture | Training | Clean Acc. | FGSM Acc. | PGD Acc. | Clean ECE | FGSM ECE |
|---|---|---|---|---|---|---|
| Small CNN | Standard | 74.96% | 6.20% | 0.29% | 0.1591 | 0.8835 |
| Small CNN | FGSM-AT | 62.00% | 35.25% | 29.13% | 0.1084 | 0.1152 |
| ResNet-18 | Standard | 81.55% | 1.73% | 0.00% | 0.1278 | 0.9477 |
| ResNet-18 | FGSM-AT | 65.42% | 31.52% | 25.15% | 0.1174 | 0.4420 |

Adversarial training consistently improves robustness against both FGSM and PGD attacks across both architectures, at the cost of reduced clean accuracy (~13 pp for Small CNN, ~16 pp for ResNet-18) and mild degradation in clean-data calibration. Standard models, while well-calibrated on clean data, collapse to near-maximum ECE under attack.

---

## Conclusion

FGSM-based adversarial training substantially improves robustness and dramatically improves calibration under attack, at the cost of a robustness tax on clean accuracy. Scaling to ResNet-18 amplifies both the benefits and costs without resolving the fundamental trade-off — larger capacity neither confers robustness automatically nor alleviates the accuracy penalty of adversarial training. Future directions include temperature scaling or other post-hoc calibration methods to recover clean-data calibration without sacrificing robustness.

---

## Repository Contents

- `Adversarial_Robustness.ipynb`: Full annotated Jupyter notebook with model definitions, training loops, attack implementations, calibration metrics, and reliability diagrams.
- `report.pdf`: Written project report with full experimental setup, quantitative results, and discussion.

---

## Dataset

CIFAR-10 is downloaded automatically via `torchvision`:

```python
datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
```

No manual setup required.

---

## Requirements

- Python 3.x
- PyTorch + torchvision
- NumPy, Pandas, Matplotlib
- Google Colab (recommended for GPU access)

---

## Authors

- Carla Aranda Sánchez 
- Jorge Barcia Belinchón
- Marina Juzgado Gómez-Menor 
- Iván López Anca
