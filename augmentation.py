# Augmenting the pose images with the synthetic backgorunds
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F # for the interpolation function
import matplotlib.pyplot as plt
import random
from PIL import Image
import PIL
import numpy as np


def random_paste(background_image, pose_image):
    w, h = pose_image.size
    start_w = 0
    start_h = 0
    resized_pose_image = pose_image # .resize((new_w, new_h))
    
    canvas_image = Image.new('RGBA', (w, h))
    canvas_image.paste(resized_pose_image, (start_w, start_h), resized_pose_image)
    background_image = background_image.copy().convert('RGBA')
    background_image.paste(resized_pose_image, (start_w, start_h), resized_pose_image)
    return background_image, canvas_image


class AddBackground:
    def __init__(self):
        self.normalize_tf = transforms.Compose([transforms.Normalize(mean=(0.5,0.5,0.5),
                                                     std=(0.5,0.5,0.5))]) # transform only for image
        
    def create_stack(self, img, mask)->None:
        # since we have a small dataset, this is feasible.
        stacked_imgs = []
        img = np.asarray(img)[:,:,:-1]
        # ref_img = np.asarray(ref_img)[:,:,:-1]
        mask = (np.asarray(mask)[:,:,-1] > 0).astype(np.float32)
        img_mask = np.concatenate((img, mask[:,:,None]), axis=-1) # stacks the mask behind the image, same augmentation needs to be applied to image
                                                                    # and mask
        stacked_imgs.append(torch.from_numpy(np.moveaxis(img_mask,-1,0)))
        images = torch.stack(stacked_imgs, dim=0)
        input_tensor = images/255. # img \in [0,1] 
        image = self.normalize_tf(input_tensor[:,:-1,:,:]) # normalizes the image
        mask = (input_tensor[:,-1, :, :] > 0).type(torch.float32)
        return image, mask
    
def test(background_images):
    '''test for the custom dataset'''
    background_add = AddBackground()
    img_tensor = background_add.create_stack(background_images[0],
                                             255*np.ones(shape=(256,256),
                                                         dtype=np.uint8))
    in_mask = img_tensor[1]
    in_tensor = img_tensor[0][0]
    in_tensor = (in_tensor - in_tensor.min()) / (in_tensor.max()- in_tensor.min())
    plt.imshow(np.moveaxis(in_tensor.numpy(),0, -1))
    plt.figure()
    plt.imshow(in_mask.numpy()[0])
    plt.show()