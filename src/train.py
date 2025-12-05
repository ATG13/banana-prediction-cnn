import sys
import os
sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
import numpy as np
from src.data_processing import load_data
from src.model import create_banana_cnn
import pickle
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping




def main():
    # Load data generators
    print("Loading data...")
    train_generator, valid_generator, test_generator = load_data()
    
    # Create model
    print("Creating model...")
    model = create_banana_cnn(num_classes=train_generator.num_classes)
    model.summary()

    # Define callbacks
    checkpoint_path = Path("../models/banana_model_best.keras")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train model
    print("Starting training...")
    epochs = 20
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=callbacks
    )

    # Save final model
    final_model_path = Path("../models/banana_model_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {checkpoint_path}")

    # Save model using pickle
    pickle_path = Path("../models/banana_model.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {pickle_path} using pickle")

if __name__ == "__main__":
    main()
