import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from modules.data_loader import PatchDatasetLoader
from modules.model import PatchMatchModel


def plot_confusion_matrix(y_true, y_pred, quantize_aware):
    """
    Generate and save confusion matrix plot for PatchMatch model on the test set.
    :param y_true: True labels array
    :param y_pred: Predicted labels array
    :param quantize_aware: Quantized model (True or False)
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    if quantize_aware == 'True':
        plt.title('PatchMatch Quantize-Aware Model Confusion Matrix on Test Set')
        plt.savefig(os.path.join('analytics/', 'PatchMatch_QA_Model_Confusion_Matrix.png'))
    else:
        plt.title('PatchMatch Model Confusion Matrix on Test Set')
        plt.savefig(os.path.join('analytics/', 'PatchMatch_Model_Confusion_Matrix.png'))
    plt.close()


def evaluate_model(device, data_dir, model_dir, quantized_model, model_confidence):
    """
    Evaluate PatchMatch model on the test set and generate relevant metrics.
    :param device: CPU or GPU
    :param data_dir: Directory containing test set
    :param model_dir: Directory containing saved model
    :param quantized_model: Quantized model (True or False)
    :param model_confidence: Model confidence threshold
    """
    # Set device for TensorFlow
    if device == 'CPU':
        tf.config.set_visible_devices([], 'GPU')
    elif device == 'GPU':
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                tf.config.set_visible_devices(physical_devices[0], 'GPU')
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found, training on CPU.")
            tf.config.set_visible_devices([], 'GPU')
    else:
        raise ValueError("Device must be 'CPU' or 'GPU'")

    # Load and preprocess data
    dataset_loader = PatchDatasetLoader(data_dir=data_dir)
    test_dataset = dataset_loader.get_test_dataset()
    print(f"[INFO] Number of test examples in dataset: {test_dataset.cardinality() * dataset_loader.batch_size}")

    print("[INFO] Loading model...")

    # Initialize model
    pm_model = PatchMatchModel()

    # Load model
    if quantized_model == 'False':
        model = pm_model.load_model(model_dir)
    else:
        model = pm_model.load_quantize_model(model_dir)

    print("[INFO] Model loaded.")
    print("[INFO] Evaluating model on the test dataset.")

    # Evaluate model on the test dataset
    test_metrics = model.evaluate(test_dataset)

    print("[INFO] Test Loss:", test_metrics[0])
    print("[INFO] Test Accuracy:", test_metrics[1] * 100)
    print("[INFO] Test Precision:", test_metrics[2] * 100)
    print("[INFO] Test Recall:", test_metrics[3] * 100)

    # Generate predictions for the confusion matrix
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((predictions > model_confidence).astype(int))

    # Plot and save the confusion matrix
    plot_confusion_matrix(y_true, y_pred, quantize_aware=quantized_model)

    print("[INFO] Test metrics analytics saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PatchMatch model.')
    parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU',
                        help='Device to train on (CPU or GPU)')
    parser.add_argument('--data_dir', type=str, default='data/dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--model_dir', type=str, default='models/saved_model/PatchMatch_Model.keras',
                        help='Path to saved model directory')
    parser.add_argument('--quantized_model', type=str, default='False',
                        help='Quantized model? (True or False)')
    parser.add_argument('--model_confidence', type=float, default=0.5, help='Model confidence threshold')

    args = parser.parse_args()
    evaluate_model(args.device, args.data_dir, args.model_dir, args.quantized_model, args.model_confidence)

