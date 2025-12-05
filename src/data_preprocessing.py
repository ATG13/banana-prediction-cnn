import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(data_dir=None, target_size=(224, 224), batch_size=32):
    """
    Creates data generators for train, validation, and test sets using ImageDataGenerator.

    Args:
        data_dir (str, optional): Path to the root 'data' directory containing 'Banana Ripeness Classification Dataset'.
                                  If None, tries to locate it relative to this script.
        target_size (tuple): Target size for the images (height, width).
        batch_size (int): Batch size for the generators.

    Returns:
        tuple: (train_generator, valid_generator, test_generator)
    """

    # Determine base data path
    if data_dir is None:
        # Assuming script is in src/ and data is in ../data/Banana Ripeness Classification Dataset
        current_script_dir = Path(__file__).resolve().parent
        base_data_path = current_script_dir.parent / 'data' / 'Banana Ripeness Classification Dataset'
    else:
        base_data_path = Path(data_dir) / 'Banana Ripeness Classification Dataset'

    if not base_data_path.exists():
        raise FileNotFoundError(f"Data directory not found at: {base_data_path}")

    print(f"Loading data from: {base_data_path}")

    # Data Augmentation for Training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Rescaling only for Validation and Test
    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create Generators
    # Using 'train' directory
    train_dir = base_data_path / 'train'
    valid_dir = base_data_path / 'valid'
    test_dir = base_data_path / 'test'

    print("Found training images:")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("Found validation images:")
    valid_generator = valid_test_datagen.flow_from_directory(
        valid_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("Found test images:")
    # Shuffle=False for test to keep order if needed for evaluation
    test_generator = valid_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False 
    )

    return train_generator, valid_generator, test_generator

if __name__ == "__main__":
    try:
        train_gen, valid_gen, test_gen = create_data_generators()
        
        # Verify a batch
        images, labels = next(train_gen)
        print(f"\nBatch Information:")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Class indices: {train_gen.class_indices}")
        
    except Exception as e:
        print(f"Error: {e}")
