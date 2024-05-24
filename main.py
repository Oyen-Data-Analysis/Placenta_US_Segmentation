import torch
from models.Attention_UNet import AttentionUNet
from torchvision.utils import make_grid
from train.train_attention_unet import train_unet
from test.test_attention_unet import test_unet

from utils.util_losses import dice_coeff, FocalLoss
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset.placenta_us_dataset import Dataset, DataLoader, Sampler

batch_size = 10
epochs = 5
optimizers = ['adam', 'rmsprop', 'adagrad']
# dataloaders = load_data()

sampler = Sampler()
dataset = Dataset(type='train', Sampler=sampler, transforms=None)
dataloaders = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

test_dataset = Dataset(type='test', Sampler=sampler, transforms=None)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)

val_dataset = Dataset(type='val', Sampler=sampler, transforms=None)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0)

def train(optim):
    model = AttentionUNet()
    if optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-5)
    criterion = FocalLoss()

    trained_model = train_unet(model, dataloaders, optimizer, criterion, num_epochs=epochs)

    return trained_model


def test(model_path):
    model = AttentionUNet()
    criterion = FocalLoss()
    dice_loss = test_unet(model, test_dataloader, criterion, model_path)

    return dice_loss


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
    best_model = models[val_loss.index(min(val_loss))]
    torch.save(best_model.state_dict(), "{0}/unet_seg_best.pth".format(save_root))

    ### test the trained model ###
    dice_loss = test(model_path=save_root + '/unet_seg_best.pth') 
    print('dice_loss: {:.4f}'.format(dice_loss)) 
 