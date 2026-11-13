
# Empirical Comparative Study of Generative Models & Theoretical Generation Barriers of Discriminative Classifiers

**Course:** Deep Learning (D6421) — Session 07: Generative Models  
**Author:** Edwin Antonie (2802397306)  
**Environment:** Python 3.10.9 | PyTorch (CUDA) | NVIDIA GeForce RTX 4060 Laptop GPU

---

## 1. Overview

This repository contains the standardized empirical implementation and evaluation of six deep generative architectures alongside a discriminative Multi-Layer Perceptron (MLP) classifier on the **MNIST** benchmark dataset:

1. **Vanilla Autoencoder (AE)** ($z = 64$)
2. **Sparse Autoencoder (Sparse AE)** ($z = 64$ with $L_1$ sparsity regularization)
3. **Variational Autoencoder (VAE)** ($z = 64$, ELBO objective)
4. **Vanilla GAN** (MLP-based Generator & Discriminator, $z = 64$)
5. **Deep Convolutional GAN (DCGAN)** (ConvTranspose2d & Conv2d, $z = 100$)
6. **Denoising Diffusion Probabilistic Model (DDPM)** (Time-conditioned U-Net/Conv backbone, $T = 300$)
7. **MLP Classifier ($784 \rightarrow 256 \rightarrow 64 \rightarrow 10$)** — Used to empirically investigate the theoretical barriers preventing discriminative classifiers from performing image generation, including the $P(y|x)$ information bottleneck, non-orthogonal weight transposition ($W^T \neq W^{-1}$) with irreversible ReLU, and unregularized latent spaces.

---

## 2. Repository Structure

```text
├── data/                                # Auto-downloaded MNIST dataset directory
├── stage1_classifier.ipynb            # Main end-to-end Jupyter Notebook
├── 1_reconstruction_grid.png            # Figure 1: Ground Truth vs AE, Sparse AE, and VAE
├── 2_generative_sampling.png            # Figure 2: Random Generative Sampling Grid
├── 3_classifier_barrier.png             # Figure 3: Empirical Proof of MLP Classifier Generation Barriers
├── 4_latent_interpolation.png            # Figure 4: Linear Latent Space Interpolation
├── 5_loss_and_trajectory.png             # Figure 5: Training Loss Curves & DDPM Denoising Trajectory
├── metrics_generative_models.csv         # Quantitative test metrics (MSE, PSNR, SSIM, Latency)
├── requirements.txt                      # Python dependencies
└── README.md                              # Execution guide and repository documentation
```

---

## 3. Experimental Standardization

To ensure a fair comparison across all architectures:

- **Dataset:** MNIST (70,000 grayscale images of size $28 \times 28$, normalized to $[-1, 1]$).
- **Data Split:** Uniform `80% Train (56,000) : 10% Validation (7,000) : 10% Test (7,000)` split using a fixed generator seed.
- **Reproducibility:** Global random seeds are fixed at `42` using `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, and `torch.cuda.manual_seed_all(42)`.
- **Training Configuration:** `BATCH_SIZE = 128`, `EPOCHS = 15` (5 epochs for MLP Classifier & Frozen Decoder), using the Adam optimizer.

---

## 4. Setup & Execution Instructions

### Step 1: Clone the Repository

Replace `<username-github-kamu>` and `<nama-repo-kamu>` with your actual GitHub username and repository name.

```bash
git clone https://github.com/<username-github-kamu>/<nama-repo-kamu>.git
cd <nama-repo-kamu>
```

### Step 2: Create & Activate Virtual Environment

```bash
python -m venv tugas7
```

**Windows (CMD):**
```cmd
tugas7\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\tugas7\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source tugas7/bin/activate
```

### Step 3: Install Dependencies

Ensure that PyTorch with CUDA support is installed if running on an NVIDIA GPU.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Note:** The CUDA-enabled PyTorch installation command above uses the CUDA 12.1 wheel index. Make sure it is compatible with your NVIDIA driver and system configuration. For CPU-only execution, install the appropriate PyTorch version instead.

### Step 4: Run the Notebook

Open `stage1_classifier_2.ipynb` in VS Code or Jupyter Notebook, select the `tugas7` kernel, and execute all cells sequentially (**Run All**).

Running all cells will:

1. Download the MNIST dataset into `./data/`.
2. Train the MLP Classifier and execute the generation-barrier experiments (Stages 1–5).
3. Train all six generative models: Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, and DDPM.
4. Export the experiment figures (`1_reconstruction_grid.png` to `5_loss_and_trajectory.png`).
5. Compute test metrics and generate `metrics_generative_models.csv`.

---

## 5. Requirements

The project uses Python libraries for deep learning, data processing, visualization, and notebook execution.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-image`
- `scikit-learn`
- `jupyter`
- `ipykernel`

To generate `requirements.txt` from the active virtual environment, run:

```bash
pip freeze > requirements.txt
```

---

## 6. Reproducibility Notes

For consistent experimental results:

- Use the same dataset and train-validation-test split across all models.
- Keep the random seed fixed at `42`.
- Maintain consistent preprocessing and image normalization.
- Use the documented training configurations when reproducing the experiments.
- Run notebook cells sequentially to ensure all required models, metrics, and figures are generated.

---

## 7. Author

**Edwin Antonie**  
Deep Learning (D6421)  
BINUS University
