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
The following architectures are explored:  
- **CNN** – Baseline architecture inspired by lensing classification studies.  
- **ResCNN** – CNN with residual connections to enhance feature propagation.  
- **ViT** – Vision Transformer for long-range spatial dependencies.  

### **Training Details**  
- **Loss Function:** Cross-Entropy  
- **Optimizer:** Adam  
- **Scheduler:** Cosine Annealing LR  
- **Data Preprocessing:** Min-max normalization (pre-applied), optional augmentation  

---

## **Usage**  

### **1️⃣ Extract Dataset**  
Convert NPY images into a PyTorch dataset:  
```bash
python scripts/data_proc.py
```

### **2️⃣ Train Model**  
Train using PyTorch or Keras. Training logs, hyperparameters, and outputs are recorded in **WandB**.  
```bash
python scripts/train.py
```
Alternatively, run the Jupyter Notebook:  
```bash
scripts/train.ipynb
```

### **3️⃣ Evaluate & Visualize**  
Generate **ROC-AUC curves** and model analysis:  
```bash
scripts/eval.ipynb
```

---

## **Results & Findings**  

### **Differences in Lens Structures**  
- **No Substructure**: Clean arcs with smooth mass distribution.  
- **Subhalo Substructure**: Distorted arcs due to small-scale gravitational perturbations.  
- **Vortex Substructure**: Complex patterns indicating unconventional gravitational effects.  

### **CNN: Standard Convolutional Approach**  
**Architecture:** Based on [Gravitational Lens Classification via CNNs](https://arxiv.org/abs/1905.04303).  
**Key Feature:** **Adaptive Average Pooling** to reduce overfitting (better than traditional max pooling).  

**Training Results:**  
- Hyperparameters & logs: `test1/logs/2025-3-17-cnn-config.yaml`  
- Training visualization:  
  ![CNN Training](images/train-cnn-2025-03-17-15432.png)  
- Final test results: ![CNN Testing](images/output-cnn.png)  

---

### **ResCNN: Residual Learning for Better Convergence**  
**Why Residual Blocks?**  
Residual connections allow gradients to flow through the network without degradation, making training deeper networks feasible.  

**Key Features:**  
- **Larger receptive fields** using **5x5** and **3x3 kernels** in residual blocks.  
- **Dropout (0.2)** to mitigate overfitting.  

**Training Insights:**  
- **ResCNN converges faster** than vanilla CNN due to skip connections.  
- **More stable gradients**, reducing vanishing/exploding issues.

**Training Results:**  
- Hyperparameters & logs: `test1/logs/2025-3-17-rescnn-config.yaml`
- Training visualization:  
  ![ResCNN Training](images/train-rescnn-2025-03-17-150214.png)  
- Final test results: ![ResCNN Testing](images/output-rescnn.png) 


**Comparison with CNN:**  
- CNN achieves **good performance but struggles with complex structures**.  
- ResCNN maintains performance while being **less prone to overfitting**.  

---

## **Next Steps**  
Experiment with **Vision Transformers (ViT)**.
Try **self-supervised pretraining** before fine-tuning on lens classification.

**Stay tuned for further model improvements!**  
