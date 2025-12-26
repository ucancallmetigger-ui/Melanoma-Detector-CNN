# Melanoma-Detector-CNN
Skin Cancer? We'll Detect It for You!

Skin Cancer Detection using Convolutional Neural Networks
This project implements an end-to-end deep learning system for binary classification of skin lesions as malignant (cancerous) or benign using real-world dermoscopic images. The solution focuses on accurate early detection of skin cancer (primarily melanoma) while maintaining strong generalization through advanced preprocessing and model optimization techniques.
Dataset & Source
The model is trained and evaluated on the publicly available "Skin Cancer: Malignant vs. Benign" dataset from Kaggle, containing over 3,300 high-quality dermoscopic images.
Classes:

Benign: ~1,800 images
Malignant: ~1,500 images (primarily melanoma and other cancerous lesions)
Images are split into train and test sets provided by the dataset, with additional validation split for monitoring.

Data Preprocessing
Key preprocessing steps include:

Automatic loading from structured folders (train/test, benign/malignant)
Resizing all images to 128x128 pixels
Normalization to [0,1] range
Extensive data augmentation for improved generalization:
Rotation (±40°)
Width/height shift (±30%)
Shear and zoom (±30%)
Horizontal and vertical flip
Brightness adjustment (0.8–1.2)

Sequence generation via ImageDataGenerator with flow_from_dataframe

Modeling Approach
A custom Convolutional Neural Network (CNN) architecture was designed and optimized for medical image classification.
Custom CNN Architecture

Conv2D layers with increasing filters (32 → 64 → 128 → 256)
Batch Normalization after each convolutional layer for training stability
MaxPooling for spatial downsampling
Dense layers (512 → 256) with ReLU activation
Dropout (0.5) for regularization
Final sigmoid output for binary classification

Training Configuration

Optimizer: Adam
Loss Function: Binary Cross-Entropy
Learning Rate: 0.001 with ReduceLROnPlateau scheduler (factor=0.5, patience=3)
Callbacks:
EarlyStopping (patience=8, restore best weights)
Automatic learning rate reduction

Epochs: Up to 50 (stopped early based on validation performance)
Batch Size: 32
Device: GPU acceleration when available

Evaluation Metrics
The model was evaluated using standard classification metrics on the held-out test set (660 images):

Overall Accuracy: 86%
Benign class: Precision 0.87 | Recall 0.86 | F1-Score 0.87
Malignant class: Precision 0.84 | Recall 0.85 | F1-Score 0.84
Macro Average F1: 0.85
Key achievement: 85% recall on malignant cases (critical for cancer detection)

Visualization & Analysis
Results include:

Training/validation accuracy and loss curves
Confusion matrix heatmap
Sample predictions with true vs predicted labels
Classification report with detailed per-class metrics
Comparative analysis highlighting balanced performance across classes

Technologies Used

Python
TensorFlow / Keras
Pandas & NumPy
Matplotlib & Seaborn
Scikit-learn
Google Colab for development and training

This project demonstrates practical application of deep learning in medical imaging, achieving strong performance suitable for portfolio showcasing, research demos, and interview discussions. The complete implementation is available as a Jupyter/Colab notebook with reproducible results.
