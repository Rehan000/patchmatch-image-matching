import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import multiprocessing
import numpy as np
import tensorflow as tf


class PatchMatch:
    """
    PatchMatch class to perform efficient, robust and fast image matching.
    """
    def __init__(self,
                 tflite_model=os.path.abspath(os.path.join('..', 'models/tflite_model/PatchMatch_TFLite.tflite')),
                 num_features=500,
                 patch_size=40,
                 match_feature='ORB',
                 k=3,
                 model_confidence=0.995):
        """
        Class constructor.
        :param tflite_model: Path to TFLite model
        :param num_features: Number of local features to detect in image
        :param patch_size: Size to the patch to extract
        :param match_feature: Local feature to detect in image
        :param k: Top k matches to consider
        :param model_confidence: Model confidence to filter predictions
        """
        self.tflite_model_path = tflite_model
        self.num_features = num_features
        self.patch_size = patch_size
        self.match_feature = match_feature
        self.k = k
        self.model_confidence = model_confidence
        self.tflite_model = self.load_tflite_model()

    def load_tflite_model(self):
        """
        Loads TFLite model.
        """
        interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path, num_threads=multiprocessing.cpu_count())
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.resize_tensor_input(input_details[0]['index'],
                                        [self.num_features * self.k, self.patch_size, self.patch_size, 1])
        interpreter.resize_tensor_input(input_details[1]['index'],
                                        [self.num_features * self.k, self.patch_size, self.patch_size, 1])
        output_shape = output_details[0]['shape']
        interpreter.resize_tensor_input(output_details[0]['index'],
                                        [self.num_features * self.k, self.patch_size, self.patch_size, 1])
        interpreter.allocate_tensors()

        return interpreter

    def detect_features(self, image):
        """
        Detects selected local feature in the given image.
        :param image: Image
        :return: Keypoints and descriptors
        """
        if self.match_feature == 'ORB':
            feature = cv2.ORB_create(nfeatures=self.num_features)
        elif self.match_feature == 'SIFT':
            feature = cv2.SIFT_create(nfeatures=self.num_features)
        elif self.match_feature == 'KAZE':
            feature = cv2.KAZE_create()
        else:
            raise TypeError("Select 'ORB', 'SIFT' or 'KAZE' as options for local features.")

        kps, des = feature.detectAndCompute(image, None)

        return kps, des

    def match_features(self, des_1, des_2):
        """
        Matches keypoints based on the features selected and their descriptors.
        :param des_1: Descriptors from image 1
        :param des_2: Descriptors from image 2
        :return: Matches
        """
        if self.match_feature == 'ORB':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif self.match_feature == 'SIFT':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif self.match_feature == 'KAZE':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise TypeError("Select 'ORB', 'SIFT' or 'KAZE' as options for local features.")

        matches = bf.knnMatch(des_1, des_2, k=self.k)

        return matches

    @staticmethod
    def extract_keypoints(kps_1, kps_2, matches):
        """
        Extracts the point coordinates from given matches.
        :param kps_1: Keypoints from image 1
        :param kps_2: Keypoints from image 2
        :param matches: Matches
        :return: Point coordinates from image 1 and image 2
        """
        kps_1_pts = [kp.pt for kp in kps_1]
        kps_2_pts = [kp.pt for kp in kps_2]
        points_1 = [kps_1_pts[m[i].queryIdx] for m in matches for i in range(len(m))]
        points_2 = [kps_2_pts[m[i].trainIdx] for m in matches for i in range(len(m))]
        points_1 = np.asarray(points_1, dtype=np.int32)
        points_2 = np.asarray(points_2, dtype=np.int32)

        return points_1, points_2

    def extract_patches(self, image_1, image_2, points_1, points_2):
        """
        Extract patches of required shape from the given coordinate points from image 1 and image 2
        :param image_1: Image 1
        :param image_2: Image 2
        :param points_1: Point coordinates from image 1
        :param points_2: Point coordinates from image 2
        :return: Patches and their point coordinates
        """
        image_1_norm = image_1 / 255.0
        image_2_norm = image_2 / 255.0

        num_points = len(points_1)
        patches_1 = np.zeros((num_points, self.patch_size, self.patch_size), dtype=np.float32)
        patches_2 = np.zeros((num_points, self.patch_size, self.patch_size), dtype=np.float32)

        valid_indices = []

        for i, (pt1, pt2) in enumerate(zip(points_1, points_2)):
            x1, y1 = max(int(pt1[0] - (self.patch_size // 2)), 0), max(int(pt1[1] - (self.patch_size // 2)), 0)
            x2, y2 = max(int(pt2[0] - (self.patch_size // 2)), 0), max(int(pt2[1] - (self.patch_size // 2)), 0)

            x1_end = min(x1 + self.patch_size, image_1.shape[1])
            y1_end = min(y1 + self.patch_size, image_1.shape[0])
            x2_end = min(x2 + self.patch_size, image_2.shape[1])
            y2_end = min(y2 + self.patch_size, image_2.shape[0])

            patch_1 = image_1_norm[y1:y1_end, x1:x1_end]
            patch_2 = image_2_norm[y2:y2_end, x2:x2_end]

            if (patch_1.shape == (self.patch_size, self.patch_size) and
                    patch_2.shape == (self.patch_size, self.patch_size)):
                patches_1[i] = patch_1
                patches_2[i] = patch_2
                valid_indices.append(i)

        patches_1 = patches_1[valid_indices]
        patches_2 = patches_2[valid_indices]

        points_1 = np.array(points_1, dtype=np.int32)[valid_indices]
        points_2 = np.array(points_2, dtype=np.int32)[valid_indices]

        return points_1, points_2, patches_1, patches_2

    def get_predictions(self, patches_1, patches_2):
        """
        Get PatchMatch model predictions for the set of patches from image 1 and image 2
        :param patches_1: Patches from image 1
        :param patches_2: Patches from image 2
        :return: Model prediction score
        """
        num_patches_1 = patches_1.shape[0]
        num_patches_2 = patches_2.shape[0]

        if num_patches_1 != num_patches_2:
            raise ValueError("patches_1 and patches_2 must have the same number of patches.")

        valid_patches_count = num_patches_1

        target = self.k * self.num_features

        if num_patches_1 < target:
            black_patches_1 = np.zeros((target - num_patches_1, self.patch_size, self.patch_size), dtype=np.float32)
            patches_1 = np.concatenate((patches_1, black_patches_1), axis=0)
            black_patches_2 = np.zeros((target - num_patches_2, self.patch_size, self.patch_size), dtype=np.float32)
            patches_2 = np.concatenate((patches_2, black_patches_2), axis=0)

        patches_1 = patches_1.reshape((patches_1.shape[0], self.patch_size, self.patch_size, 1))
        patches_2 = patches_2.reshape((patches_2.shape[0], self.patch_size, self.patch_size, 1))

        input_details = self.tflite_model.get_input_details()
        output_details = self.tflite_model.get_output_details()
        self.tflite_model.set_tensor(input_details[0]['index'], patches_1)
        self.tflite_model.set_tensor(input_details[1]['index'], patches_2)
        self.tflite_model.invoke()
        predictions = self.tflite_model.get_tensor(output_details[0]['index']).squeeze()

        if valid_patches_count < target:
            predictions = predictions[:valid_patches_count]

        return predictions

    def filter_predictions(self, points_1, points_2, predictions):
        """
        Filter PatchMatch model predictions based on the given confidence score
        :param points_1: Point coordinates from image 1
        :param points_2: Point coordinates from image 2
        :param predictions: Model prediction scores
        :return: Filtered point coordinates from image 1 and image 2
        """
        filter_indices = np.where(predictions >= self.model_confidence)[0]
        model_filtered_points_1 = points_1[filter_indices]
        model_filtered_points_2 = points_2[filter_indices]

        return model_filtered_points_1, model_filtered_points_2

    def match_two_images(self, image_1, image_2):
        """
        Perform image matching using the PatchMatch pipeline.
        :param image_1: Image 1
        :param image_2: Image 2
        :return: Corresponding point coordinates from image 1 and image 2
        """
        image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
        image_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

        kps_1, des_1 = self.detect_features(image_gray_1)
        kps_2, des_2 = self.detect_features(image_gray_2)

        matches = self.match_features(des_1, des_2)

        points_1, points_2 = self.extract_keypoints(kps_1, kps_2, matches)

        points_1, points_2, patches_1, patches_2 = self.extract_patches(image_gray_1, image_gray_2, points_1, points_2)

        predictions = self.get_predictions(patches_1, patches_2)

        points_1, points_2 = self.filter_predictions(points_1, points_2, predictions)

        return points_1, points_2
