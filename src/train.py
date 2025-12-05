import sys
import os
sys.path.append(os.path.abspath('../'))

import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import create_banana_cnn 

def train_banana_model(data_dir='../data/', epochs=50, batch_size=32):
    """
    Trains CNN model for banana ripeness classification.
    
    Args:
        data_dir: Path to data directory with train/val/test folders
        epochs: Maximum training epochs
        batch_size: Batch size for training
    
    Saves:
        models/bestmodel.h5: Best model by validation accuracy
        models/traininghistory.pkl: Training history
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data from directory structure
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'valid'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Create and compile model
    model = create_banana_cnn()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/bestmodel.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    with open('../models/traininghistory.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"Training completed. Best model saved as 'models/bestmodel.h5'")
    print(f"Training history saved as 'models/traininghistory.pkl'")
    return model, history

if __name__ == "__main__":
    model, history = train_banana_model()
