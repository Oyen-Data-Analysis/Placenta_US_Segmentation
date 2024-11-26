import numpy as np
import pydicom
import os
import SimpleITK as sitk
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.distributed import DistributedSampler
import random
from utils.util_image import *
import matplotlib.pyplot as plt

# import config # use this import if running placenta_us_dataset.py
import dataset.config as config
from utils.util_image import center_crop

# Patient1 = Patient('104-1', os.path.join(config.FGR_PATH, 'export1/project/Dataset/Placenta_OCT/FGR/104-1'))
# visit = ['104-1/Visit 2/IMG_20230727_1_30.mha']
# Patient1.add_visit(visit)

SEGMENTATION_PATH = {'FGR': config.FGR_PATH, 'Controlled': config.CONTROLLED_PATH}
MASK_PATH = config.MASK_PATH

def load_us_dataset(
        # indexes,
        # root_path: str = '/export1/project/Dataset/Placenta_OCT/',
        verbose: bool = False):
    
    # scan_file_list = ["C:\\Users\\DRACula\\Box\\In Utero Wellcome Leap Project\\FGR Study Ultrasound Data\\Placenta Data\\Analysis\\FGR_Patients_Segmented/104-1/Visit 2/IMG_20230727_1_30.dcm",
    #                   "C:\\Users\\DRACula\\Box\\In Utero Wellcome Leap Project\\FGR Study Ultrasound Data\\Placenta Data\\Analysis\\Control_Patients_Segmented/114-1/Visit 2/IMG_20230811_1_34.dcm"]
    # segmented_file_list = ["C:\\Users\\DRACula\\Box\\In Utero Wellcome Leap Project\\FGR Study Ultrasound Data\\Placenta Data\\Analysis\\FGR_Patients_Segmented/104-1/Visit 2/IMG_20230727_1_30.mha",
    #                        "C:\\Users\\DRACula\\Box\\In Utero Wellcome Leap Project\\FGR Study Ultrasound Data\\Placenta Data\\Analysis\\Control_Patients_Segmented/114-1/Visit 2/IMG_20230811_1_34.mha"]
    scan_file_list = []
    segmented_file_list = []
    if verbose:
        print("RUNNING load_us_dataset ...")
        print("========================================================")
    for path in SEGMENTATION_PATH.values():
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".dcm"):
                    f = root + "/" + f
                    if MASK_PATH == '':
                        mask_path = f.replace(".dcm", ".mha")
                    else:
                        mask_path = MASK_PATH + '/' + os.path.basename(f).replace('.dcm', '_segmented.jpg')
                    if os.path.exists(mask_path):
                        scan_file_list.append(f)
                        segmented_file_list.append(mask_path)

    return scan_file_list, segmented_file_list

class Sampler:
    def __init__(self, proportions=[0.8, 0.1, 0.1]):
        self.data_input, self.data_labels = load_us_dataset()
        self.proportions = proportions
        self.train, self.val, self.test = self.split_data(proportions)

    def split_data(self, proportions):
        train = []
        val = []
        test = []
        for i in range(len(self.data_input)):
            rand = random.random()
            if rand < proportions[0]:
                train.append(i)
            elif rand < proportions[0] + proportions[1]:
                val.append(i)
            else:
                test.append(i)
        return train, val, test
    
    def get_train(self):
        return [self.data_input[i] for i in self.train], [self.data_labels[i] for i in self.train]
    
    def get_val(self):
        return [self.data_input[i] for i in self.val], [self.data_labels[i] for i in self.val]
    
    def get_test(self):
        return [self.data_input[i] for i in self.test], [self.data_labels[i] for i in self.test]
    
    def get_debug(self):
        arr1 = [os.path.join(config.FGR_PATH, '139-1/Visit 2/IMG_20231027_1_52.dcm'), os.path.join(config.FGR_PATH, '132-1/Visit 2/IMG_20231023_1_58.dcm'), os.path.join(config.FGR_PATH, '136-1/Visit 2/IMG_20231019_2_41.dcm')]
        arr2 = [os.path.join(config.FGR_PATH, '139-1/Visit 2/IMG_20231027_1_52.mha'), os.path.join(config.FGR_PATH, '132-1/Visit 2/IMG_20231023_1_58.mha'), os.path.join(config.FGR_PATH, '136-1/Visit 2/IMG_20231019_2_41.mha')]
        return arr1, arr2
    
    def get_all(self):
        return self.data_input, self.data_labels

class Dataset:
    def __init__(
            self,
            type,
            Sampler,
            # root_path='C:\\Users\\DRACula\\Box\\In Utero Wellcome Leap Project\\FGR Study Ultrasound Data\\Placenta Data\\Analysis\\',
            transforms=None
    ):
        self.transforms = transforms

        self.sampler=Sampler
        self.type = type
        self.data_input, self.data_labels = self.sampler.get_train() if self.type == 'train' else self.sampler.get_val() if self.type == 'val' else self.sampler.get_test() if self.type == 'test' else self.sampler.get_debug()      
        # self.data_input, self.data_labels = self.sampler.get_all() if self.type == 'train' else ([], [])

    def __len__(self):

        return len(self.data_input)

    def __getitem__(self, item):
        # index_subject, index_z = self.indexes_map[item]
        index_subject = self.data_input[item]
        img_input = read_dicom_image(index_subject)

        # Convert Input to Grayscale and reshape
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_input = img_input.reshape(1, img_input.shape[0], img_input.shape[1])


        index_subject_label = self.data_labels[item]
        if index_subject_label.endswith('.mha'):
            img_label = read_mha_image(index_subject_label)
            cnts = cv2.findContours(img_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                cv2.drawContours(img_label, [c], 0, (255, 255, 255), -1)
        else:
            img_label = cv2.imread(index_subject_label, cv2.IMREAD_GRAYSCALE)

        img_label = img_label.reshape(1, img_label.shape[0], img_label.shape[1])

        img_label = np.where(img_label > 0, 1, 0)

        img_input = center_crop(img_input, [600, 600])
        img_label = center_crop(img_label, [600, 600])

        img_input = cv2.normalize(img_input, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return img_input, img_label
    
    def returnTrainingSet(self):
        return Dataset()

# def load_data_for_diffusion_model(is_train = True, is_distributed = False, num_workers=0):
#     # dataset = Dataset(indexes=[1, 2, 4, 5, 6], transforms=None)
#     dataset = Dataset(indexes=[1], transforms=None)


#     if is_train:
#         data_sampler = None
#         if is_distributed:
#             data_sampler = DistributedSampler(dataset)
#         loader = DataLoader(
#             dataset,
#             batch_size=5,
#             shuffle=(data_sampler is None) and is_train,
#             sampler=data_sampler,
#             num_workers=num_workers,
#             drop_last=is_train,
#             pin_memory=True,
#         )
#         # return loader
#         while True:
#             yield from loader

#     else:
#         data_sampler = None
#         # if is_distributed:
#         #     data_sampler = DistributedSampler(dataset)
#         loader = DataLoader(
#             dataset,
#             batch_size=5,
#             shuffle=False,
#             sampler=data_sampler,
#             num_workers=num_workers,
#             drop_last=is_train,
#             pin_memory=True,
#     )
#         # return loader
#         while True:
#             yield from loader

def read_dicom_image(path):
    """Reads and returns a DICOM image as a numpy array."""
    dicom_image = pydicom.dcmread(path)
    image_array = dicom_image.pixel_array
    image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0  # Normalize
    image_array = np.uint8(image_array)
    return image_array

def read_mha_image(path):
    """Reads and returns an MHA image as a numpy array."""
    itk_image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(itk_image)
    image_array = np.squeeze(image_array)  # Remove singleton dimensions if any
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = image_array[:, :, 0]
        print("3 Channels. Taking first. Shape is now: ", image_array.shape)
    return np.uint8(image_array)

if __name__ == '__main__':
    # a, b = load_us_dataset(verbose=True)
    # indexes_map = []
    # for index_subject in a:
    #     for index_z in range(
    #             0, 0
    #                + 100, 1
    #     ):
    #         indexes_map.append([index_subject, index_z])
    # print(len(indexes_map))
    sampler = Sampler()
    dataset = Dataset(type='train', Sampler=sampler, transforms=None)
    print(dataset.__len__())
    img_array = dataset.__getitem__(0)[0]
    if np.all(img_array == 0):
        print("Image is all zeros")
    print(img_array)
    trainloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    # for i, data in enumerate(trainloader):  # inner loop within one epoch
    #     input, label = data
    #     print(input.shape)
    #     print(label.shape)
    #     print(i)