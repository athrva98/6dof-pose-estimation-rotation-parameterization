# training loop
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from model import NeuralNet, rotationHead
from torch.utils.data import DataLoader
from dataloader import _6dofDataset
import matplotlib.pyplot as plt
from loss_functions import LOSS_FUNCTION
import os


torch.manual_seed(111)
from visualize_rotation import visualizeAngVel6, visualizeRotMat12, visualizequat7, visualizeaffine16
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def visualize_mask(mask, mask_hat, image, idx, path):
    # plt.ion()
    fig, axs = plt.subplots(2)
    # print(image.shape)
    mask = mask.detach().cpu().numpy()
    mask_hat = mask_hat.detach().cpu().numpy()
    axs[0].cla()
    axs[1].cla()
    axs[0].imshow(mask[0][0])
    axs[1].imshow(mask_hat[0][0])
    axs[0].axis('off')
    axs[1].axis('off')
    # plt.savefig(path + os.path.sep + f'{idx}.png', bbox_inches='tight')
    plt.close()
    # plt.close()
    # plt.draw()
    # plt.pause(0.0001)




def train_loop(dataloader, num_epochs, learning_rate, load=False, path=None, save=False):

    gamma = 30/2.5
    backbone = NeuralNet(in_channels=3, out_channels=1).to(device)
    rotation_head = rotationHead(_type=0).to(device)

    # Training
    optimizer = torch.optim.Adam(list(backbone.parameters())+list(rotation_head.parameters()),
                         lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=3)
    criterion = torch.nn.BCELoss() # we will use the binary cross entropy loss.

    start = 0
    if load:
        print("Loading checkpoint.....")
        checkpoint = torch.load(path)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        rotation_head.load_state_dict(checkpoint['roation_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_load = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Checkpoint loaded at epoch {} and loss {}".format(epoch_load, loss))
        start = epoch_load
    loss_curve = []
    for epoch in range(start, num_epochs):

        # Training phase
        backbone.train()
        rotation_head.train()
        running_loss = 0.
        # running_val_loss = 0.
        idx_ctr = 0
        with tqdm(dataloader, unit="batch") as tepoch:
          for (imgtensor, pose) in tepoch:
              idx_ctr += 1
              image = imgtensor[0][0]
              mask = imgtensor[1]
              mask_hat, pose_logits = backbone.forward(image.to(device))
              pose_logits = rotation_head(pose_logits)
              optimizer.zero_grad()
              l1 = criterion(torch.sigmoid(mask_hat)[0][0], mask.to(device)[0][0])
              l2 =  gamma * LOSS_FUNCTION.loss_RotMat12(pose_logits, pose.to(device), gamma=2.5)

              loss = (l1 + l2)
              loss.backward()
              optimizer.step()
              running_loss += loss.item()

              # Empty RAM and GPU
            #   visualize_mask(mask, mask_hat, image, idx=idx_ctr, path='./RotMat12_masks')
            #   visualizeaffine16(pose_logits, pose, idx=idx_ctr, path='./affine16_orientations')
              torch.cuda.empty_cache()
        scheduler.step()
        train_loss = running_loss/idx_ctr # len(dataloader)
        loss_curve.append(train_loss)
        path = './models_affine16/model_checkpoint_'+str(epoch+1)
        if save:
            torch.save({
                'epoch': epoch+1,
                'backbone_state_dict': backbone.state_dict(),
                'roation_head_state_dict': rotation_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
                }, path)


        print('\nEpoch: {}, Training Loss: {:.4f}'.format(epoch+1, train_loss))
    return backbone, rotation_head


def train():
    dataset = _6dofDataset()
    dataloader = DataLoader(dataset, batch_size=1)
    
    model = train_loop(dataloader, num_epochs=107, learning_rate=3e-3, load=False,
                         path='./models_RotMat12/model_checkpoint_101')


def check_create_folders():
    paths = ['models_RotMat12', 'models_AngVel6', 'models_quat7', 'models_affine16']
    for k in range(len(paths)):
        if not os.path.exists('./' + paths[k]):
            os.mkdir('./' + paths[k])

if __name__ == '__main__':
    check_create_folders()
    train()