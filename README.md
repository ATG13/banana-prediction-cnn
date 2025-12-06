# Banana Ripeness Prediction CNN

A computer vision project to classify the ripeness of bananas using a Convolutional Neural Network (CNN). This project provides tools for training a model, evaluating its performance, and deploying it via a user-friendly Streamlit web interface.

## Project Overview

The goal of this project is to accurately categorize bananas into different ripeness stages (e.g., Unripe, Ripe, Overripe, Rotten) based on their images.

## Directory Structure

```
├── app/
│   └── app.py              # Streamlit web application for real-time inference
├── data/                   # Directory to store the dataset
├── models/                 # Directory to save trained model artifacts
├── notebooks/              # Jupyter notebooks for data exploration and prototyping
├── src/
│   ├── data_processing.py  # Utilities for data loading and preprocessing
│   ├── model.py            # Definition of the CNN model architecture
│   ├── train.py            # Script to train the model
│   └── test.py             # Script to evaluate the model on test data
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup and Installation

1.  **Clone the repository** (if not already done).

2.  **Install Dependencies**:
    Ensure you have Python installed. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model
To train the CNN model on your dataset, run the training script:
```bash
python src/train.py
```
This will process the data, train the model, and save the best performing model to the `models/` directory.

### 2. Testing the Model
To evaluate the trained model's performance on the test set:
```bash
python src/test.py
```
This runs the model against the test data and outputs performance metrics.

### 3. Running the Web Application
To launch the interactive web interface where you can upload images for classification:
```bash
streamlit run app/app.py
```
Open the provided local URL in your web browser to interact with the app.

## Technologies Used

*   **TensorFlow/Keras**: For building and training the deep learning model.
*   **Streamlit**: For creating the web application frontend.
*   **OpenCV & Pillow**: For image processing.
*   **NumPy & Pandas**: For numerical operations and data handling.
*   **Matplotlib & Seaborn**: For data visualization.
