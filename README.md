# Lung Cancer Detection Using Deep Learning

A deep learning system for automated lung cancer classification from histopathological images using transfer learning with VGG16.

## Project Overview

This project implements a computer vision system to classify lung histopathological images into three categories:

- **Adenocarcinoma (lung_aca)** - The most common type of lung cancer
- **Squamous Cell Carcinoma (lung_scc)** - The second most common type
- **Normal/Benign Tissue (lung_n)** - Healthy lung tissue

The model is designed to assist pathologists in diagnosing lung cancer more efficiently, with potential applications in early detection and screening.

## Model Performance

The model achieves approximately 98% accuracy on the test set. Here's a detailed breakdown:

| Metric | Adenocarcinoma | Normal Tissue | Squamous Cell | Overall |
|--------|----------------|---------------|---------------|---------|
| **Precision** | 95% | 100% | 99% | 98% |
| **Recall** | 99% | 100% | 95% | 98% |
| **F1-Score** | 97% | 100% | 97% | 98% |

**Test Set Performance:** 2,940 correct predictions out of 3,000 images

### Training Process

The model was trained using a two-stage approach:

- **Stage 1 (Epochs 1-2):** Training only the classification head achieved 96.17% validation accuracy
- **Stage 2 (Epochs 3-5):** Fine-tuning the last three VGG16 layers improved validation accuracy to 98.12%
- **Early Stopping:** Automatically triggered at epoch 5, with best weights restored from epoch 4

## Architecture

The model uses transfer learning with VGG16 pre-trained on ImageNet:

```
Input: 224x224x3 RGB images
    |
VGG16 Base (Pre-trained on ImageNet)
  - Layers 0-15: Frozen to preserve universal features
  - Layers 16-18: Fine-tuned for histopathology-specific features
    |
Flatten Layer
    |
Dropout (0.5) - Regularization
    |
Dense Layer (128 neurons, ReLU activation)
    |
Dropout (0.3) - Additional regularization
    |
Output Layer (3 neurons, Softmax activation)
```

**Total Parameters:** 17.9 million  
**Trainable Parameters:** 10.3 million

## Dataset

**Source:** [LC25000 Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

- **Total Images:** 15,000 lung histopathology images
- **Original Size:** 768×768 pixels (resized to 224×224 for training)
- **Training Split:** 80% (12,000 images)
- **Testing Split:** 20% (3,000 images)
- **Class Distribution:** Balanced - 5,000 images per class
- **Staining Method:** H&E (Hematoxylin and Eosin)
- **Magnification:** 40× objective (400× total magnification)

## Getting Started

### Prerequisites

```
Python 3.8 or higher
TensorFlow 2.x
NumPy
Pillow
Matplotlib
scikit-learn
seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hvsssen/cnn_vgg16_lung_cancer_detection.git
cd cnn_vgg16_lung_cancer_detection
```

2. Install the required dependencies:
```bash
pip install tensorflow numpy pillow matplotlib scikit-learn seaborn
```

3. Download the dataset using the Kaggle API:
```bash
kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images
```

### Training the Model

Open the Jupyter notebook and execute the cells in order:

```bash
jupyter notebook cnn_vgg16_lung_cancer_detection.ipynb
```

**Training Configuration:**
- Learning Rate: 0.0001 (Stage 1), 0.00001 (Stage 2)
- Batch Size: 16
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Regularization: Dropout layers (0.5 and 0.3) plus data augmentation
- Early Stopping: Patience of 3 epochs

## Data Augmentation

To prevent overfitting and account for real-world variations in histopathological imaging:

- **Brightness Adjustment:** 0.8-1.2× (simulates different lighting conditions)
- **Contrast Variation:** 0.8-1.2× (simulates different staining intensities)
- **Normalization:** All pixel values scaled to [0, 1]

These augmentations help the model generalize better to images from different laboratories and microscopes.

## Progressive Unfreezing Strategy

The training uses a two-stage approach that has proven effective for medical image classification:

**Stage 1: Classification Head Training**
- All VGG16 layers remain frozen
- Only the custom classification layers are trained
- This prevents random gradients from disrupting the pre-trained weights
- The new layers stabilize from their random initialization

**Stage 2: Fine-Tuning**
- The last three VGG16 convolutional layers are unfrozen
- These layers adapt to histopathology-specific features
- Lower learning rate prevents catastrophic forgetting
- Results in improved accuracy compared to training without fine-tuning

This approach typically yields 2-3% better accuracy than either freezing all layers or training everything from scratch.

## Making Predictions

After training, you can use the model to classify new histopathological images:

```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Make a prediction
image_path = 'path/to/new_image.jpeg'
detect_and_display(image_path, model)
```

The inference function will display the image along with the predicted class and confidence score.

## Technologies Used

- **Deep Learning Framework:** TensorFlow/Keras
- **Transfer Learning Base:** VGG16 (ImageNet pre-trained weights)
- **Image Processing:** NumPy, Pillow
- **Data Visualization:** Matplotlib, Seaborn
- **Model Evaluation:** scikit-learn
- **Development Environment:** Google Colab with GPU acceleration

## Project Structure

```
cnn_vgg16_lung_cancer_detection/
├── cnn_vgg16_lung_cancer_detection.ipynb    # Main training notebook
├── model.h5                                  # Trained model weights
├── README.md                                 # Project documentation
└── Read/                                     # Additional documentation
```

## Key Learnings

This project demonstrates several important concepts in deep learning and medical image analysis:

- Effective use of transfer learning for medical imaging tasks
- Progressive unfreezing as a fine-tuning strategy
- Data augmentation techniques for histopathological images
- Handling imbalanced medical datasets
- Model evaluation metrics appropriate for healthcare applications
- Production considerations for deploying medical AI systems

## Future Work

Potential improvements and extensions:

- Implement Grad-CAM visualization to highlight regions influencing the model's decisions
- Explore ensemble methods combining multiple architectures
- Develop a web interface for easier clinical integration
- Optimize inference speed for real-time applications
- Validate performance on external datasets from different institutions
- Extend to multi-task learning (classification + localization)

## Important Disclaimer

**This project is for research and educational purposes only.**

This model has not been validated for clinical use and should not be used for medical diagnosis without proper validation and regulatory approval. It is intended as a proof-of-concept and learning tool. Any clinical application would require:

- Extensive validation by certified pathologists
- Regulatory approval (FDA, CE marking, etc.)
- Integration with established clinical workflows
- Ongoing monitoring and quality assurance

Always consult qualified healthcare professionals for medical decisions.

## Acknowledgments

**Dataset:**  
Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. "Lung and Colon Cancer Histopathological Image Dataset (LC25000)." arXiv:1912.12142v1 [eess.IV], 2019.

**VGG16 Architecture:**  
Simonyan, K. & Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR 2015.

**Computing Resources:**  
This project was developed using Google Colab's free GPU resources.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

**Hassen**
- GitHub: [@hvsssen](https://github.com/hvsssen)
- Email: hvsssen7@gmail.com

## Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue on this repository
- Email: hvsssen7@gmail.com

---

**Developed as part of exploring deep learning applications in medical imaging**
