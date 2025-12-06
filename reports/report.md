# Banana Ripeness Classification Report

## Introduction
This report presents the development and evaluation of a Convolutional Neural Network (CNN) designed to classify the ripeness stages of bananas. The goal of the project is to accurately categorize bananas into four distinct classes (e.g., Unripe, Ripe, Overripe, Rotten) based on visual data, facilitating automated quality control and sorting.

## Methodology

### Data Preprocessing
The image dataset was processed using TensorFlow's `ImageDataGenerator`. All images were resized to a target dimension of **224x224 pixels** and normalized by rescaling pixel values to the [0, 1] range (1./255).

To improve model generalization and robustness, the training data underwent real-time data augmentation with the following parameters:
*   **Rotation**: Up to 20 degrees
*   **Width/Height Shifts**: Up to 20%
*   **Horizontal Flip**: Enabled
*   **Fill Mode**: Nearest

Validation and test sets were rescaled but not augmented to ensure a fair evaluation.

### Model Architecture
The model utilizes a sequential CNN architecture comprising:
1.  **Convolutional Blocks**: Three blocks, each containing a `Conv2D` layer (starting with 32 filters, increasing to 64) with ReLU activation, followed by `MaxPooling2D` for spatial downsampling.
2.  **Classification Head**: A `Flatten` layer converts the 2D feature maps into a 1D vector, fed into a fully connected `Dense` layer with 128 units (ReLU activation). A `Dropout` layer (rate 0.5) is included to prevent overfitting, followed by a final `Dense` layer with Softmax activation for class probability distribution.

### Training Configuration
*   **Optimizer**: Adam
*   **Loss Function**: Categorical Crossentropy
*   **Callbacks**: `ModelCheckpoint` to save the best performing weights based on validation accuracy, and `EarlyStopping` to halt training if validation loss did not improve for 5 consecutive epochs.

## Results and Evaluation
The model was trained and evaluated on the split dataset. The performance metrics for the best-saved model are as follows:

*   **Validation Accuracy**: 96.44%
*   **Test Accuracy**: 95.55%
*   **Test Loss**: 0.1181

These high accuracy scores indicate that the model has successfully learned to distinguish distinctive features of banana ripeness and generalizes well to unseen data.

## Conclusion
The project successfully implemented a CNN achieving approximately **95.6% accuracy** on the test set. The breakdown of methodology highlights a robust approach using data augmentation and a standard yet effective CNN architecture. The results demonstrate the feasibility of using computer vision for automated fruit ripeness assessment. Future improvements could explore transfer learning with larger architectures (e.g., MobileNet, ResNet) to potentially further boost accuracy or efficiency for edge deployment.
