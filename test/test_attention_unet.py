import copy
import time
import torch
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.nn import functional as F
from utils.util_image import *

def test_unet(model, dataloaders, criterion, model_path):
    since = time.time()
    # Use gpu if available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(test_state_dict)

    # Initialize batch summary
    batch_test_loss = 0.0
    batch_test_psnr = 0.0


    model.eval()

    # Iterate over data.
    label_list = []
    outputs_list = []
    input_list = []
    for i, data in enumerate(dataloaders):
        input, label = data

        inputs = input.float().to(device)
        label = label.float().to(device)

        # track history only in training phase
        outputs = model(inputs)
        outputs = (outputs > 0.55).to(outputs.dtype)

        loss = criterion(outputs, label).item()

        # plt.imshow(outputs[0, 0].detach().cpu(), cmap='gray')
        # plt.show()
        # plt.imshow(label[0, 0].detach().cpu(), cmap='gray')
        # plt.show()

        # accumulate batch loss
        batch_test_loss += loss
        # print(outputs.shape)
        outputs_list.append(outputs.squeeze())
        label_list.append(label.squeeze())
        input_list.append(input.squeeze())

    # save epoch losses
    epoch_test_loss = batch_test_loss / len(dataloaders.dataset.data_input)

    out_tif = torch.stack(outputs_list, dim=0)
    out_tif = torch.flatten(out_tif, start_dim=0, end_dim=1).detach().cpu().numpy()
    to_tiff(x=out_tif, path='C:/Users/DRACula/Documents/Research/Oyen Lab/US-segmentation-project-Chen/results/06_outputs.tif',
            is_normalized=False)

    label_tif = torch.stack(label_list, dim=0)
    label_tif = torch.flatten(label_tif, start_dim=0, end_dim=1).detach().cpu().numpy()
    to_tiff(x=label_tif, path='C:/Users/DRACula/Documents/Research/Oyen Lab/US-segmentation-project-Chen/results/06_labels.tif',
            is_normalized=False)


    input_tif = torch.stack(input_list, dim=0)
    input_tif = torch.flatten(input_tif, start_dim=0, end_dim=1).detach().cpu().numpy()
    to_tiff(x=input_tif, path='C:/Users/DRACula/Documents/Research/Oyen Lab/US-segmentation-project-Chen/results/06_inputs.tif',
            is_normalized=False)

    return epoch_test_loss