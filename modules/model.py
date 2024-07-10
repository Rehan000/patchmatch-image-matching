import tensorflow as tf
import tensorflow_model_optimization as tfmot


class PatchMatchModel:
    """
    PatchMatch model class for efficient and robust image matching.
    """
    def __init__(self, h=40, w=40, c=1):
        """
        Class constructor.
        :param h: Patch height in pixels, default = 40
        :param w: Patch width in pixels, default = 40
        :param c: Patch channels, default = 1
        """
        self.H = h
        self.W = w
        self.C = c
        self.QuantizeModel = tfmot.quantization.keras.quantize_model
        self.Converter = tf.lite.TFLiteConverter.from_keras_model
        self.model = self.build_model()
        self.quantized_model = None
        self.tflite_model = None

    def build_model(self):
        """
        Define model architecture and return PatchMatch model.
        :return: model
        """
        patch_1 = tf.keras.layers.Input(shape=[self.H, self.W, self.C], name='patch_1')
        patch_2 = tf.keras.layers.Input(shape=[self.H, self.W, self.C], name='patch_2')

        x = tf.keras.layers.concatenate([patch_1, patch_2])

        down1 = tf.keras.layers.Conv2D(filters=8,
                                       kernel_size=4,
                                       strides=2,
                                       padding='same',
                                       kernel_initializer='random_normal',
                                       use_bias=False)(x)
        down1 = tf.keras.layers.LeakyReLU()(down1)

        down2 = tf.keras.layers.Conv2D(filters=16,
                                       kernel_size=4,
                                       strides=2,
                                       padding='same',
                                       kernel_initializer='random_normal',
                                       use_bias=False)(down1)
        down2 = tf.keras.layers.BatchNormalization()(down2)
        down2 = tf.keras.layers.LeakyReLU()(down2)

        down3 = tf.keras.layers.Conv2D(filters=32,
                                       kernel_size=3,
                                       strides=2,
                                       padding='same',
                                       kernel_initializer='random_normal',
                                       use_bias=False)(down2)
        down3 = tf.keras.layers.BatchNormalization()(down3)
        down3 = tf.keras.layers.LeakyReLU()(down3)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv = tf.keras.layers.Conv2D(filters=32,
                                      kernel_size=3,
                                      strides=1,
                                      kernel_initializer='random_normal',
                                      use_bias=False)(zero_pad1)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        flatten = tf.keras.layers.Flatten()(leaky_relu)

        dense_1 = tf.keras.layers.Dense(units=8, activation='relu')(flatten)

        output = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense_1)

        model = tf.keras.Model(inputs=[patch_1, patch_2], outputs=output)

        return model

    def get_model(self):
        """
        Returns PatchMatch model.
        :return: model
        """
        return self.model

    def load_model(self, model_dir):
        """
        Load model from saved model checkpoint or saved model file.
        :param model_dir: Path to saved model checkpoint or saved model
        :return: model
        """
        self.model = tf.keras.models.load_model(model_dir)

        return self.model

    def get_quantize_model(self):
        """
        Quantize aware model for quantize aware training.
        :return: quantized_model
        """
        self.quantized_model = self.QuantizeModel(self.model)
        return self.quantized_model

    def load_quantize_model(self, quantize_model_dir):
        """
        Load quantize-aware model from saved model checkpoint or saved model file.
        :param quantize_model_dir: Path to saved quantize-aware model checkpoint or saved quantize-aware model
        :return: quantized_model
        """
        with tfmot.quantization.keras.quantize_scope():
            self.quantized_model = tf.keras.models.load_model(quantize_model_dir)

        return self.quantized_model

    def get_tflite_model(self):
        """
        Convert quantized model to tflite model.
        :return: tflite_model
        """
        if self.quantized_model is not None:
            converter = self.Converter(self.quantized_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            self.tflite_model = converter.convert()

        return self.tflite_model

    def save_tflite_model(self, tflite_model_dir):
        """
        Save quantized PatchMatch TFLite model to the given directory.
        :param tflite_model_dir: TFLite model directory
        """
        with open(tflite_model_dir, 'wb') as f:
            f.write(self.tflite_model)
