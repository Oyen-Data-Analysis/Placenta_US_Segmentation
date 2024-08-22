import torch
from models.Attention_UNet import AttentionUNet
from test.test_attention_unet import test_unet
from utils.util_losses import FocalLoss
from dataset.placenta_us_dataset import Dataset, DataLoader, Sampler


if __name__ == '__main__':
    model_path = 'model_zoo/unet_seg_adam.pth'
    model = AttentionUNet()
    Sampler = Sampler()
    test_dataloader = DataLoader(Dataset(type='debug', Sampler=Sampler, transforms=None), batch_size=1, shuffle=False, num_workers=0)
    criterion = FocalLoss()

    print(test_unet(model, test_dataloader, criterion, model_path))