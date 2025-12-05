import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


def load_data():
    """
    Docstring for load_data
    """
    # Paths to pre-split directories
    BASE_DIR = Path("../data")
    TRAIN_DIR = BASE_DIR / "train"
    VAL_DIR = BASE_DIR / "valid"
    TEST_DIR = BASE_DIR / "test"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Training with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


    # Validation/Test with only normalization
    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    # Load from pre-split directories
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = valid_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = valid_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, valid_generator, test_generator