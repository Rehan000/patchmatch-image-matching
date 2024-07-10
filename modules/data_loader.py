import os
import tensorflow as tf


class PatchDatasetLoader:
    """
    Dataloader class for PatchMatch model training, validation and testing.
    """
    def __init__(self, data_dir, batch_size=32):
        """
        Class constructor.
        :param data_dir: Data directory containing training, validation and test datasets
        :param batch_size: Batch size, default = 32
        """
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_image_paths = tf.data.Dataset.list_files(os.path.join(data_dir, "train", "*", "*.jpg"))
        self.valid_image_paths = tf.data.Dataset.list_files(os.path.join(data_dir, "valid", "*", "*.jpg"))
        self.test_image_paths = tf.data.Dataset.list_files(os.path.join(data_dir, "test", "*", "*.jpg"))

        self.train_dataset = self._prepare_dataset(self.train_image_paths, is_training=True)
        self.valid_dataset = self._prepare_dataset(self.valid_image_paths, is_training=False)
        self.test_dataset = self._prepare_dataset(self.test_image_paths, is_training=False)

    @staticmethod
    def _load(image_file):
        """
        Load patches image from directory, split into two patches and return.
        :param image_file: Patches image file
        :return: return both patches
        """
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)

        w = tf.shape(image)[1] // 2
        patch_1 = image[:, :w, :]
        patch_2 = image[:, w:, :]

        patch_1 = tf.cast(patch_1, tf.float32)
        patch_2 = tf.cast(patch_2, tf.float32)

        return patch_1, patch_2

    @staticmethod
    def _normalize(patch_1, patch_2):
        """
        Normalize both patches.
        :param patch_1: Patch 1
        :param patch_2: Patch 2
        :return: return both patches
        """
        patch_1 = patch_1 / 255.
        patch_2 = patch_2 / 255.

        return patch_1, patch_2

    def _load_image(self, image_file):
        """
        Load and pre-process patches images.
        :param image_file: Patches image file
        :return: return both patches
        """
        patch_1, patch_2 = self._load(image_file)
        patch_1, patch_2 = self._normalize(patch_1, patch_2)

        return patch_1, patch_2

    @staticmethod
    def _get_label(image_path):
        """
        Get labels for the loaded patches.
        :param image_path:
        :return: Label
        """
        image_path = tf.strings.split(image_path, "/")
        contains_good_match = tf.reduce_any(tf.equal(image_path, "good_match"))
        label = tf.cond(contains_good_match, lambda: 1, lambda: 0)

        return label

    def _prepare_dataset(self, image_paths, is_training):
        """
        Prepare and return dataset.
        :param image_paths: Image paths
        :param is_training: True or False depending on if dataset is training
        :return: Dataset
        """
        dataset = image_paths.map(lambda x: (self._load_image(x), self._get_label(x)))
        if is_training:
            dataset = dataset.shuffle(buffer_size=10000).batch(self.batch_size)
        else:
            dataset = dataset.batch(self.batch_size)
        return dataset

    def get_train_dataset(self):
        """
        Returns training dataset.
        :return: Train dataset
        """
        return self.train_dataset

    def get_valid_dataset(self):
        """
        Returns validation dataset.
        :return: Valid dataset
        """
        return self.valid_dataset

    def get_test_dataset(self):
        """
        Returns test dataset.
        :return: Test dataset
        """
        return self.test_dataset
