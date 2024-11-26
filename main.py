import torch
from models.Attention_UNet import AttentionUNet
from torchvision.utils import make_grid
from train.train_attention_unet import train_unet
from test.test_attention_unet import test_unet

from utils.util_losses import dice_coeff, FocalLoss
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset.placenta_us_dataset import Dataset, DataLoader, Sampler

def train(optim):
    model = AttentionUNet()
    device = torch.device("cuda:1")
    model.to(device)
    if optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-5)
    else:
        print("Invalid optimizer")
        return None
    criterion = FocalLoss()

    trained_model = train_unet(model, dataloaders, optimizer, criterion, num_epochs=epochs)

    return trained_model

def test(model_path):
    model = AttentionUNet()
    criterion = FocalLoss()
    dice_loss = test_unet(model, test_dataloader, criterion, model_path)

    return dice_loss

batch_size = 2
val_test_batch_size = 1
epochs = 5
# optimizers = ['adam']
optimizers = ['adam', 'rmsprop', 'adagrad']
# dataloaders = load_data()

sampler = Sampler()
dataset = Dataset(type='train', Sampler=sampler, transforms=None)
dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
print(len(dataloaders.dataset.data_input))


test_dataset = Dataset(type='test', Sampler=sampler, transforms=None)
test_dataloader = DataLoader(test_dataset, batch_size=val_test_batch_size, shuffle=False, num_workers=0)
print(len(test_dataloader.dataset.data_input))


val_dataset = Dataset(type='val', Sampler=sampler, transforms=None)
val_dataloader = DataLoader(val_dataset, batch_size=val_test_batch_size, shuffle=False, num_workers=0)
print(len(val_dataloader.dataset.data_input))


if __name__ == '__main__':
    models = []
    val_loss = []
    save_root = './model_zoo'
    for optim in optimizers:
        trained_model = train(optim)[0]
        models.append(trained_model)        
        torch.save(trained_model.state_dict(), "{0}/unet_seg_{1}.pth".format(save_root, optim))
        val_loss.append(test_unet(trained_model, val_dataloader, FocalLoss(), "{0}/unet_seg_{1}.pth".format(save_root, optim)))
        print('Validation loss for {0}: {1}'.format(optim, val_loss))
    best_model_idx = val_loss.index(min(val_loss))
    best_model = models[best_model_idx]
    best_optim = optimizers[best_model_idx]
    torch.save(best_model.state_dict(), "{0}/unet_seg_best_{1}.pth".format(save_root, best_optim))

    ### test the trained model ###
    dice_loss = test(model_path=save_root + '/unet_seg_best_{0}.pth'.format(best_optim))
    print('dice_loss: {:.4f}'.format(dice_loss))
    print('used {0} optimizer'.format(best_optim))