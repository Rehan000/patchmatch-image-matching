import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from modules.data_loader import PatchDatasetLoader
from modules.model import PatchMatchModel


def plot_and_save_metrics(history, metric_name, qa_training):
    """
    Generate and save analytics plots for tracking model loss, accuracy, precision and recall.
    :param history: Model training history
    :param metric_name: Metric to be plotted
    :param qa_training: Metrics for quantized model training or normal model training (True or False)
    """
    plt.figure()
    plt.plot(history.history[metric_name], label=f'Training {metric_name}')
    plt.plot(history.history[f'val_{metric_name}'], label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if qa_training:
        plt.title(f'PatchMatch Quantize-Aware Model Training and Validation {metric_name.capitalize()}')
    else:
        plt.title(f'PatchMatch Model Training and Validation {metric_name.capitalize()}')
    plt.legend()
    plt.grid()
    if qa_training:
        plot_path = os.path.join('analytics/', f'PatchMatch_QA_Model_{metric_name.capitalize()}.png')
    else:
        plot_path = os.path.join('analytics/', f'PatchMatch_Model_{metric_name.capitalize()}.png')
    plt.savefig(plot_path)
    plt.close()


def train_model(data_dir, device, epochs, checkpoint_dir, continue_training, quantize_aware_training,
                checkpoint_dir_quant):
    """
    Train PatchMatch model and track training and validation metrics.
    :param data_dir: Training and validation dataset directory
    :param device: CPU or GPU
    :param epochs: Number of epochs for training
    :param checkpoint_dir: Checkpoint directory for model
    :param continue_training: Continue training from saved checkpoint
    :param quantize_aware_training: Train quantize aware model
    :param checkpoint_dir_quant: Checkpoint directory for quantized model
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
    train_dataset = dataset_loader.get_train_dataset()
    valid_dataset = dataset_loader.get_valid_dataset()
    print(f"[INFO] Number of training examples in dataset: {train_dataset.cardinality() * dataset_loader.batch_size}")
    print(f"[INFO] Number of validation examples in dataset: {valid_dataset.cardinality() * dataset_loader.batch_size}")

    # Initialize model
    pm_model = PatchMatchModel()
    if continue_training == 'False':
        model = pm_model.get_model()
    else:
        model = pm_model.load_model(checkpoint_dir)

    if quantize_aware_training == 'True':
        _ = pm_model.load_model(checkpoint_dir)
        model = pm_model.get_quantize_model()

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    # Set callback to save model checkpoints
    if quantize_aware_training == 'False':
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            verbose=1,
            save_weights_only=False,
            save_freq='epoch'
        )

        # Train the model
        history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=[cp_callback])

        # Generate and save training metrics' plots
        plot_and_save_metrics(history, metric_name='loss', qa_training=False)
        plot_and_save_metrics(history, metric_name='accuracy', qa_training=False)
        plot_and_save_metrics(history, metric_name='precision', qa_training=False)
        plot_and_save_metrics(history, metric_name='recall', qa_training=False)
        print("[INFO] Saved model training metrics in analytics folder.")

    else:
        # Set callback to save model checkpoints
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir_quant,
            verbose=1,
            save_weights_only=False,
            save_freq='epoch'
        )

        # Train the model
        history = model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=[cp_callback])

        # Generate and save training metrics' plots
        plot_and_save_metrics(history, metric_name='loss', qa_training=True)
        plot_and_save_metrics(history, metric_name='accuracy', qa_training=True)
        plot_and_save_metrics(history, metric_name='precision', qa_training=True)
        plot_and_save_metrics(history, metric_name='recall', qa_training=True)
        print("[INFO] Saved model training metrics in analytics folder.")

    # Save model
    if quantize_aware_training == 'False':
        model.save('models/saved_model/PatchMatch_Model.keras')
        print("[INFO] Saved model to saved_model directory.")
    else:
        _ = pm_model.get_tflite_model()
        pm_model.save_tflite_model('models/tflite_model/PatchMatch_TFLite.tflite')
        print("[INFO] Saved TFLite model to tflite_model directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PatchMatch model.')
    parser.add_argument('--data_dir', type=str, default='data/dataset', help='Path to the dataset directory')
    parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU',
                        help='Device to train on (CPU or GPU)')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train the model')
    parser.add_argument('--checkpoint_dir', type=str, default='models/model_checkpoint',
                        help='Path to save model checkpoint')
    parser.add_argument('--continue_training', type=str, default='False',
                        help='Continue model training from model checkpoint (True or False)')
    parser.add_argument('--quantize_aware_training', type=str, default='False',
                        help='Quantize aware model training (True or False)')
    parser.add_argument('--checkpoint_dir_quant', type=str,
                        default='models/quantize_aware_model_checkpoint',
                        help='Path to save quantize aware model checkpoint')

    args = parser.parse_args()
    train_model(args.data_dir, args.device, args.epochs, args.checkpoint_dir, args.continue_training,
                args.quantize_aware_training, args.checkpoint_dir_quant)

