# **Specific Test IV: Diffusion Models**  

## **Task**  
Develop a generative model to simulate realistic strong gravitational lensing images. Train a diffusion model (DDPM)[https://arxiv.org/abs/2006.11239] to generate lensing images. You are encouraged to explore various architectures and implementations within the diffusion model framework. Please implement your approach in PyTorch or Keras and discuss your strategy.


## **Dataset Description**  
**[dataset.zip](https://drive.google.com/file/d/1cJyPQzVOzsCZQctNBuHCqxHnOY7v7UiA/view?usp=sharing)**
The dataset comprises 10,000 strong lensing images.

## **Evaluation Metrics**  
Use qualitative assessments and quantitative metrics such as Fréchet Inception Distance (FID) to measure the realism and variety of the generated samples.

---

## **Approach & Strategy**  

### **Model Selection**  
Explored architectures include:  
- **UNet** – Efficient at capturing local spatial details through convolutional layers.
  - **With self-attention** – Enhances modeling by explicitly capturing long-range dependencies and global image structures.
  - **Without self-attention** – Effective locally but struggles to represent global structures, causing blurry or less coherent outputs.

Loss functions considered:  
- **MSE (Mean Squared Error)** – Minimizes pixel-wise errors, effective for sharpness but often results in blurry images due to averaging.
- **SSIM (Structural Similarity Index)** – Optimizes perceptual similarity, preserving global structure and texture patterns.

### **Training Details**  
- **Optimizer:** Adam  
- **Scheduler:** Cosine Annealing LR (optional, improves convergence and avoids local minima)

---

## **Usage**  

### **Extract Dataset**  
Convert `.npy` gravitational lens images into a PyTorch dataset:  
```bash
python scripts/data_proc.py
```
or use the notebook:
```bash
scripts/notebooks/data_proc.ipynb
```

### **Train Model**  
Training progress, hyperparameters, and outputs are logged using **Weights & Biases (WandB)**:  
```bash
python scripts/train.py
```

### **Evaluate & Visualize**  
Compute **FID Score** and analyze generated samples:
```bash
scripts/eval.ipynb
```

---

## **Results & Findings**  

### **Observations from Sample Data**  
- Gravitational lens images typically exhibit subtle arcs, rings, and complex global structures.
- Maintaining sharpness of lensing patterns is crucial; global coherence significantly affects visual realism.

### **UNet (CNN-only with MSE Loss)**  
Pure convolutional UNet struggles to capture global patterns, resulting in clear local structures but incoherent or blurred overall shapes.

**Training Results:**  
- Hyperparameters & logs: `test4/logs/2025-3-25-wo-att-mse-config.yaml`  
- Analysis details: see `scripts/eval.ipynb`

---

### **UNet with Self-Attention**  

**Why self-attention?**  
Self-attention explicitly models global dependencies across the image, addressing the CNN's limitations in representing spatially distant structures. It allows the network to effectively capture global lens structures.

**Key Benefits:**  
- Improved representation of global patterns like arcs and rings.
- More coherent structural details in generated images.

#### Choice of Loss Function: MSE vs. MSE + SSIM  
- **Pure MSE:** Stable and faster convergence but produces less visually convincing global structures.
- **MSE + SSIM:** Better maintains perceptual quality and structural coherence, though validation loss experiences more fluctuations during training.

Both approaches reach comparable final losses (~0.003) after 100 epochs. Using SSIM is recommended for improved perceptual realism.

---

## **Next Steps**  
- Experiment with advanced perceptual losses (e.g., VGG perceptual loss).
- Further hyperparameter tuning (learning rates, scheduler parameters).
- Explore conditional diffusion models to control generation quality explicitly.
- Expand dataset size and augmentations to enhance generalization.
