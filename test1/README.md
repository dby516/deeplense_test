# Common Test I: Multi-Class Classification  

## Task  
Develop a model to classify images into three lensing categories using **PyTorch** or **Keras**. Choose the most appropriate approach and explain your strategy.  

## Dataset Description  
📂 **dataset.zip** (Google Drive)  
The dataset contains images belonging to three classes:  
- **No Substructure** (Strong lensing without substructure)  
- **Subhalo Substructure**  
- **Vortex Substructure**  

- Images are already **min-max normalized**, but further normalization and data augmentation are allowed.  

## Evaluation Metrics  
📈 **Metrics:**  
- **ROC Curve** (Receiver Operating Characteristic)  
- **AUC Score** (Area Under the ROC Curve)  



## Strategy  
🔹 **Model Selection:** CNN-based architectures (e.g., ResNet, EfficientNet)  
🔹 **Loss Function:** Categorical Cross-Entropy  
🔹 **Optimizer:** Adam / SGD  
🔹 **Augmentations:** Random flips, rotations, and intensity shifts  
🔹 **Normalization:** Standardization or further min-max scaling  

## Usage  
1. **Extract** the dataset  
2. **Train** the model using PyTorch/Keras  
3. **Evaluate** using ROC-AUC  

🚀 **Goal:** Achieve high AUC while maintaining generalization!