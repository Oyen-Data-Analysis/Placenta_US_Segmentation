# import h5py
import numpy as np
# from glob import glob
import pydicom
# from tqdm import tqdm
import os
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.distributed import DistributedSampler
import random
# from utils.util_image import *
import config # use this import if running placenta_us_dataset.py
from patient import Patient
# from dataset.patient import Patient # use this import if running main.py
# import dataset.config as config


PATIENT2FILE_MAP = {}

SEGMENTATION_PATH = {'FGR': config.FGR_PATH, 'Controlled': config.CONTROLLED_PATH}

def load_us_dataset(
        # indexes,
        # root_path: str = '/export1/project/Dataset/Placenta_OCT/',
        verbose: bool = False):
    
    scan_file_list = []
    segmented_file_list = []
    if verbose:
        print("RUNNING load_oct_dataset ...")
        print("========================================================")
    for path in SEGMENTATION_PATH.values():
        dir_list = os.listdir(path)
        for patient_dir_path in dir_list: # dir_path is id
            PATIENT2FILE_MAP[patient_dir_path] = Patient(patient_dir_path, path + '/' + patient_dir_path)
            for visit_dir_path in os.listdir(path + '/' + patient_dir_path):
                visit = []
                for file in os.listdir(path + '/' + patient_dir_path + '/' + visit_dir_path):
                    if file.endswith(".dcm"):
                        scan_id = file.rstrip('.dcm')
                        if os.path.exists(path + '/' + patient_dir_path + '/' + visit_dir_path + '/' + scan_id + '.mha'):
                            shared_path = patient_dir_path + '/' + visit_dir_path + '/' + scan_id # Append this path to one of the paths in SEGENMENTATION_PATH and append with '.dcm' or '.mha' to get the original/segmented image
                            visit.append(shared_path)
                            scan_file_list.append(path + '/' + shared_path + '.dcm')
                            segmented_file_list.append(path + '/' + shared_path + '.mha')
                PATIENT2FILE_MAP[patient_dir_path].add_visit(visit)
    
    # for index in indexes:
    #     file_id = INDEX2FILE_MAP[index]
    #     file_name = root_path + file_id + '.tif'
    #     print(file_name)
    #     file_list.append(file_name)

    if verbose:
        visit_indent = 5
        file_indent = 5

        print("returning read file structure ...")
        print("========================================================")

        for idx, (patient_id, patient) in enumerate(PATIENT2FILE_MAP.items()):
            print(f'Patient {idx + 1}: {patient_id}')
            for visit_idx, visit in enumerate(patient.visits):
                print('-' * visit_indent + f'Visit {visit_idx + 1}')
                for file in visit:
                    print(' ' * (visit_indent - 3) + '-' * file_indent + file)
        print("========================================================")
        print('\n')
        print('returning files lists ...')
        print("========================================================")
        for idx, scan in enumerate(scan_file_list):
            print(f'Scan {idx + 1}: {scan}')
            print(f'Segmented {idx + 1}: {segmented_file_list[idx]}')
            print("------------------------------------------------------------")
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
        self.data_input, self.data_labels = self.sampler.get_train() if self.type == 'train' else self.sampler.get_val() if self.type == 'val' else self.sampler.get_test()        

        # 3D Data
        # for index_subject in self.data_input:
        #     for index_z in range(
        #             0, 0
        #                + 100, 1
        #     ):
        #         self.indexes_map.append([index_subject, index_z])

        # for index_subject in self.data_input:
        #     self.indexes_map.append(index_subject)


    def __len__(self):

        return len(self.data_input)

    def __getitem__(self, item):
        # index_subject, index_z = self.indexes_map[item]
        index_subject = self.data_input[item]
        img_input = read_dicom_image(index_subject)
        index_subject_label = self.data_labels[item]
        img_label = read_mha_image(index_subject_label)

        # img_input = np.expand_dims(center_crop(img_input, [248, 248]), axis=0)
        # img_label = np.expand_dims(center_crop(img_label, [248, 248]), axis=0)

        # img_input = normlize(img_input)

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
    image_array = dicom_image.pixel_array.astype(float)
    image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0  # Normalize
    image_array = np.uint8(image_array)
    return image_array

def read_mha_image(path):
    """Reads and returns an MHA image as a numpy array."""
    itk_image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(itk_image)
    image_array = np.squeeze(image_array)  # Remove singleton dimensions if any
    return image_array

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
    print(dataset.__getitem__(0))
    trainloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    for i, data in enumerate(trainloader):  # inner loop within one epoch
        input, label = data
        print(input.shape)
        print(label.shape)
        print(i)