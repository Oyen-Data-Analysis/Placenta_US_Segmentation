import copy
import time

import math
import torch
import numpy as np
import os
import csv
from utils.util_losses import dice_coeff, FocalLoss
import matplotlib.pyplot as plt
from torchvision.utils import make_grid



import copy
import time
import torch
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.nn import functional as F


def train_unet(model, dataloaders, optimizer, criterion, num_epochs=1):
    since = time.time()
    # Use gpu if available

    train_epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        # Initialize batch summary
        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs}')
        batch_train_loss = 0.0

        device = torch.device("cuda:1")
        model.to(device)

        model.train()

        # Iterate over data.
        for i, data in enumerate(dataloaders):
            input, label = data

            inputs = input.float().to(device)
            # print("input sent to device ", inputs.device)
            label = label.float().to(device)
            # print("label sent to device ", label.device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # track history only in training phase
            outputs = model(inputs)

            loss = criterion(outputs, label)

            # back-propagation
            loss.backward()
            optimizer.step()

            # accumulate batch loss
            batch_train_loss += loss.item() * input.size(0)

            # if i % 5 == 0:
            #     print(next(model.parameters()))
            # if math.isnan(batch_train_loss):
            #     print("loss.item() is {:.4f}".format(loss.item()))
            #     print("input size is {}".format(input.size(0)))
            #     print("outputs are ")
            #     print(outputs)
            #     print("labels are ")
            #     print(label)
            #
            # print(batch_train_loss)
        # save epoch losses
        epoch_train_loss = batch_train_loss / len(dataloaders)
        train_epoch_losses.append(epoch_train_loss)

        print('Loss: {:.4f}'.format(epoch_train_loss))

    return model, train_epoch_losses

#
# def train_and_test(model, dataloaders, optimizer, criterion, num_epochs=3, show_images=False):
#     since = time.time()
#     best_loss = 1e10
#     # Use gpu if available
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     fieldnames = ['epoch', 'training_loss', 'test_loss', 'training_dice_coeff', 'test_dice_coeff']
#     train_epoch_losses = []
#     test_epoch_losses = []
#     for epoch in range(1, num_epochs + 1):
#
#         print(f'Epoch {epoch}/{num_epochs}')
#         print('-' * 10)
#         # Each epoch has a training and validation phase
#         # Initialize batch summary
#         batchsummary = {a: [0] for a in fieldnames}
#         batch_train_loss = 0.0
#         batch_test_loss = 0.0
#
#         # for phase in ['training', 'test']:
#         for phase in ['training']:
#
#             if phase == 'training':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()  # Set model to evaluate mode
#
#             # Iterate over data.
#             for i, data in enumerate(dataloaders):
#                 # print(i)
#                 input, label = data
#                 if show_images:
#                     grid_img = make_grid(input)
#                     grid_img = grid_img.permute(1, 2, 0)
#                     plt.imshow(grid_img)
#                     plt.show()
#
#                 inputs = input.float().to(device)
#                 masks = label.float().to(device)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # track history only in training phase
#                 with torch.set_grad_enabled(phase == 'training'):
#                     outputs = model(inputs)
#
#                     loss = criterion(outputs, masks)
#
#                     y_pred = outputs.data.cpu().numpy().ravel()
#                     y_true = masks.data.cpu().numpy().ravel()
#
#                     batchsummary[f'{phase}_dice_coeff'].append(dice_coeff(y_pred, y_true))
#
#                     # back-propagation
#                     if phase == 'training':
#                         loss.backward()
#                         optimizer.step()
#
#                         # accumulate batch loss
#                         batch_train_loss += loss.item() * input.size(0)
#
#                     else:
#                         batch_test_loss += loss.item() * input.size(0)
#
#             # save epoch losses
#             if phase == 'training':
#                 # epoch_train_loss = batch_train_loss / len(dataloaders['training'])
#                 epoch_train_loss = batch_train_loss / len(dataloaders)
#                 train_epoch_losses.append(epoch_train_loss)
#             else:
#                 epoch_test_loss = batch_test_loss / len(dataloaders['test'])
#                 test_epoch_losses.append(epoch_test_loss)
#
#             batchsummary['epoch'] = epoch
#             # batchsummary[f'{phase}_loss'] = epoch_train_loss.item()
#             print('{} Loss: {:.4f}'.format(phase, loss))
#
#         best_loss = np.max(batchsummary['test_dice_coeff'])
#         for field in fieldnames[3:]:
#             batchsummary[field] = np.mean(batchsummary[field])
#         print(
#             f'\t\t\t train_dice_coeff: {batchsummary["training_dice_coeff"]}, test_dice_coeff: {batchsummary["test_dice_coeff"]}')
#
#     # summary
#     print('Best dice coefficient: {:4f}'.format(best_loss))
#
#     return model, train_epoch_losses, test_epoch_losses