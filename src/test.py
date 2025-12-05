import sys
import os
from pathlib import Path

# Add project root to path to import src modules correctly
# This assumes the script is run from the src directory or we can resolve the path relative to this file
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from tensorflow.keras.models import load_model
from src.data_processing import load_data

def test_main():
    # Load best model
    # Assuming models are in models/ directory at the project root
    model_path = project_root / "models" / "banana_model_best.keras"
    
    # data_processing.py uses relative path "../data", so we must ensure CWD is src/
    # This allows the script to be run from project root or src/
    if Path.cwd().name != 'src':
        print(f"Changing working directory to {current_file.parent} to support data loading...")
        os.chdir(current_file.parent)

    # Load data
    print("Loading data...")
    train_generator, valid_generator, test_generator = load_data()
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Evaluate on Train Data
    print("\n" + "="*30)
    print("Evaluating on Train Data...")
    print("="*30)
    train_loss, train_acc = model.evaluate(train_generator)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")

    # Evaluate on Validation Data
    print("\n" + "="*30)
    print("Evaluating on Validation Data...")
    print("="*30)
    val_loss, val_acc = model.evaluate(valid_generator)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Evaluate on Test Data
    # Although the user asked specifically for train and val, it's usually good to show test too, 
    # or arguably test is what 'test.py' implies.
    print("\n" + "="*30)
    print("Evaluating on Test Data...")
    print("="*30)
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    test_main()
