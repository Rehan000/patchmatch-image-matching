import os
import random
import cv2
import numpy as np


class GeneratePatches:
    """
    GeneratePatches class to generate training, validation and test datasets
    for PatchMatch model using HPatches dataset.
    """
    def __init__(self, folder_path,
                 save_folder_path,
                 dataset_size=1000,
                 patch_size=40,
                 x_offset_low=20,
                 x_offset_high=40,
                 y_offset_low=20,
                 y_offset_high=40):
        """
        Class constructor.
        :param folder_path: Folder path for HPatches dataset
        :param save_folder_path: Folder path for saved training, validation and test dataset
        :param dataset_size: Generated dataset size
        :param patch_size: Patch crop size, default = 40
        :param x_offset_low: X-axis lower offset limit for bad match, default = 20
        :param x_offset_high: X-axis higher offset limit for bad match, default = 40
        :param y_offset_low: Y-axis lower offset limit for bad match, default = 20
        :param y_offset_high: Y_axis lower offset limit for bad match, default = 40
        """
        self.dataset_size = dataset_size
        self.patch_size = patch_size
        self.x_offset_low = x_offset_low
        self.x_offset_high = x_offset_high
        self.y_offset_low = y_offset_low
        self.y_offset_high = y_offset_high
        self.save_folder_path = save_folder_path
        self.folder_path = folder_path
        self.data_dict = self.load_images_and_homographies()

    def load_images_and_homographies(self):
        """
        Load HPatches dataset into a dictionary accessible by their set names.
        :return: data_dict
        """
        data_dict = {}
        subdirectories = [d for d in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, d))]

        for subdirectory in subdirectories:
            subdirectory_path = os.path.join(self.folder_path, subdirectory)

            images = []
            homographies = {}

            for i in range(1, 7):
                image_path = os.path.join(subdirectory_path, f"{i}.ppm")
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(image)

            for i in range(1, 6):
                homography_path = os.path.join(subdirectory_path, f"H_1_{i + 1}")
                with open(homography_path, 'r') as f:
                    lines = f.readlines()
                    homography = [list(map(float, line.strip().split())) for line in lines]
                    homographies[i - 1] = np.array(homography)

            data_dict[subdirectory] = {
                "images": images,
                "homographies": homographies
            }

        return data_dict

    def generate_data(self, data_type, dataset_size=None):
        """
        Generates dataset for PatchMatch model depending on the type selected: train, valid, test, generates equal
        number of good and bad matches.
        :param data_type: train, valid or test indicating the respective directory to save data
        :param dataset_size: Total number of examples to save, half as good matches and half as bad matches
        """
        if dataset_size is not None:
            self.dataset_size = dataset_size
        check_count = 0
        loop = True
        while loop:
            for key in self.data_dict.keys():
                for i in range(5):
                    image_1 = self.data_dict[key]['images'][0]
                    image_2 = self.data_dict[key]['images'][i + 1]
                    homography = self.data_dict[key]['homographies'][i]
                    if check_count < self.dataset_size / 2:
                        check = self._generate_match(image_1, image_2, homography, data_type, match_type='good')
                        if check:
                            check_count += 1
                    else:
                        check = self._generate_match(image_1, image_2, homography, data_type, match_type='bad')
                        if check:
                            check_count += 1
                    if check_count >= self.dataset_size:
                        loop = False
                        break
                if not loop:
                    break

    def _generate_match(self, image1, image2, homography_matrix, data_type, match_type):
        """
        Select point in image 1 and finds corresponding point in image 2 and generate good or bad match patches.
        :param image1: Image 1
        :param image2: Image 2
        :param homography_matrix: Homography between image 1 and image 2
        :param data_type: train, valid or test, indicates directory to save data
        :param match_type: good or bad match indicating match type
        :return: True or False indicating whether match is generated and returned
        """
        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]

        while True:
            point1 = np.array([np.random.randint(0, width1), np.random.randint(0, height1), 1])
            point2 = np.dot(homography_matrix, point1)
            point2 /= point2[2]

            if 0 <= point2[0] < width2 and 0 <= point2[1] < height2:
                point2 = np.round(point2).astype(int)
                break

        if match_type == 'good':
            return self._generate_good_match(image1, image2, point1, point2, data_type)

        elif match_type == 'bad':
            return self._generate_bad_match(image1, image2, point1, point2, data_type)

    def _generate_good_match(self, image1, image2, point1, point2, data_type):
        """
        Generates a valid/good match.
        :param image1: Image 1
        :param image2: Image 2
        :param point1: Point coordinates in image 1
        :param point2: Corresponding point coordinated in image 2
        :param data_type: train, valid or test
        :return: True or False indicating whether a match is generated and returned
        """
        patch_1 = self._extract_patch(image1, point1)
        patch_2 = self._extract_patch(image2, point2)

        if patch_1.shape == (self.patch_size, self.patch_size) and patch_2.shape == (self.patch_size, self.patch_size):
            combined_patch = np.hstack((patch_1, patch_2))
            self._save_patches(combined_patch, data_type, 'good_match')
            return True

        return False

    def _generate_bad_match(self, image1, image2, point1, point2, data_type):
        """
        Generates an invalid/bad match.
        :param image1: Image 1
        :param image2: Image 2
        :param point1: Point coordinates in image 1
        :param point2: Corresponding point coordinated in image 2
        :param data_type: train, valid or test
        :return: True or False indicating whether a match is generated and returned
        """
        x_offset = random.randint(self.x_offset_low, self.x_offset_high)
        y_offset = random.randint(self.y_offset_low, self.y_offset_high)

        point2[0] += x_offset if random.random() < 0.5 else -x_offset
        point2[1] += y_offset if random.random() < 0.5 else -y_offset

        patch_1 = self._extract_patch(image1, point1)
        patch_2 = self._extract_patch(image2, point2)

        if patch_1.shape == (self.patch_size, self.patch_size) and patch_2.shape == (self.patch_size, self.patch_size):
            combined_patch = np.hstack((patch_1, patch_2))
            self._save_patches(combined_patch, data_type, 'bad_match')
            return True

        return False

    def _extract_patch(self, image, point):
        """
        Given an image and point coordinates, extract a patch of required size.
        :param image: Image
        :param point: Point coordinates
        :return: patch
        """
        top_left_x = point[0] - self.patch_size // 2
        top_left_y = point[1] - self.patch_size // 2
        bottom_right_x = top_left_x + self.patch_size
        bottom_right_y = top_left_y + self.patch_size

        return image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    def _save_patches(self, patch, data_type, match_type):
        """
        Save patches to the appropriate directory
        :param patch: Patches combined
        :param data_type: train, valid or test
        :param match_type: good or bad match
        """
        filename = f"{match_type}_{str(random.randint(0, 1000000000))}.jpg"
        filepath = os.path.join(self.save_folder_path, data_type, match_type, filename)
        cv2.imwrite(filepath, patch)
