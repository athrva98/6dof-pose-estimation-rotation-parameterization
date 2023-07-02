# Writing the dataloader

# 6 dof pose estimation from rgb images
from typing import Tuple, List
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader
from augmentation import AddBackground, random_paste
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
import random

DATASET_PATH = '/home/athrva/Documents/pose_estimation_dataset/dataset_super_reduced/'

def byfilenum(path):
  num = int(path.split(os.path.sep)[-1].split('.')[0])
  return num

class _6dofDataset(Dataset):
  def __init__(self, sub_sample_length=-1)->None:
    self.backgroundAdd = AddBackground() # invokes the BigGAN to add background
    # print(self.synthetic_dataset)
    ALL_FOLDERS = glob(DATASET_PATH + os.path.sep + '/*') # this contains all the different objects
    # print(ALL_FOLDERS)
    self.rgb_images = []
    self.poses = []
    self.NUM_FOLDERS = len(ALL_FOLDERS)
    self.ridx = 0
    for FOLDER in ALL_FOLDERS:
      self.rgb_images.append(sorted(glob(FOLDER + os.path.sep + 'rgb/*.png'), key=byfilenum))
      self.poses.append(sorted(glob(FOLDER + os.path.sep + 'pose/*.txt'), key=byfilenum))
      if not sub_sample_length == -1:
        sample_idxs = np.random.randint(low=0, high=len(self.rgb_images[-1]), size=(sub_sample_length,))
        self.rgb_images[-1] = self.rgb_images[-1][sample_idxs]
        self.poses[-1] = self.poses[-1][sample_idxs]
      
  
  def __getitem__(self, idx: int)->Tuple:
    if self.ridx == self.NUM_FOLDERS-1:
      self.ridx = 0 # reset if all the folders are done
    ridx = self.ridx
    bimg = (np.ones(shape=(256, 256, 3)) * 255).astype(np.uint8)
    ref_image = cv2.imread(self.rgb_images[ridx][0])
    image = cv2.imread(self.rgb_images[ridx][idx])
    ref_image = cv2.resize(ref_image, (256, 256), cv2.INTER_CUBIC)
    image = cv2.resize(image, (256, 256), cv2.INTER_CUBIC)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGBA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    blank = np.where(cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY) >= 230)
    image[:,:,-1][blank] = 0
    blank = np.where(cv2.cvtColor(ref_image, cv2.COLOR_RGBA2GRAY) >= 230)
    ref_image[:,:,-1][blank] = 0
    canvas = Image.fromarray(bimg)
    image, mask = random_paste(canvas, Image.fromarray(image))
    ref_image, ref_mask = random_paste(canvas, Image.fromarray(ref_image))
    # print(np.asarray(self.synthetic_dataset[sidx]).shape, image.shape)
    image = Image.fromarray(np.asarray(image) - np.asarray(ref_image)) # calculates the photometric difference between images
    mask = Image.fromarray(np.asarray(mask) - np.asarray(ref_mask)) 
    img_tensor = self.backgroundAdd.create_stack(image, mask)
    with open(self.poses[ridx][idx], 'r') as pose_file:
      pose_raw = pose_file.readlines()
    pose = torch.from_numpy(np.array([float(eval(pose_raw[m])) for m in range(len(pose_raw))], dtype=np.float32))

    with open(self.poses[ridx][0], 'r') as pose_file:
      ref_pose_raw = pose_file.readlines()
    ref_pose = torch.from_numpy(np.array([float(eval(ref_pose_raw[m])) for m in range(len(ref_pose_raw))], dtype=np.float32))
    ref_pose_inv = torch.inverse(ref_pose.reshape(4,4))
    pose = (ref_pose_inv @ pose.reshape(4,4)).flatten()

    if idx >= 249:
      self.ridx += 1 # increment the folder counter
    return (img_tensor, pose)
  
  def __len__(self):
    return len(self.rgb_images[0])

def test():
  plt.ion()
  fig, axs = plt.subplots(2)
  dobj = _6dofDataset()
  for k in trange(len(dobj)):
    sample = dobj.__getitem__(k)
    print(sample[0][0].shape, sample[0][1].shape, sample[1].shape)
    print(sample[0][1][0].numpy().min(), sample[0][1][0].numpy().max())

    # plt.cla()
    # axs[0].imshow(torch.movedim(sample[0][0][0], 0, -1).numpy())
    # axs[1].imshow(sample[0][1][0].numpy())
    # plt.draw()
    # plt.pause(0.00001)


  # print(sample)

if __name__ == '__main__':
  test()






