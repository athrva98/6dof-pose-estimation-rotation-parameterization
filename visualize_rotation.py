# Functions for visualizing rotations under different parameterizations
import torch
import matplotlib.pyplot as plt
import numpy as np
from loss_functions import _skew, _quaternion_to_rotation
import os

# plt.ion()



def visualizeRotMat12(logits, true, idx, path):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    true_rot = torch.cat((true[:,:3].reshape(1,3), true[:,4:7].reshape(1,3), true[:,8:11].reshape(1,3)),dim=0)
    rotation_pred = logits[:,:9].view(3,3).detach().cpu()
    u, s, vt = np.linalg.svd(rotation_pred.numpy())
    rotation_pred = u @ vt
    if np.linalg.det(rotation_pred) < 0:
        rotation_pred = -1 * rotation_pred
    X, Y, Z = np.meshgrid(np.linspace(0, 0.25, 15)-0.125, np.linspace(0, 0.25, 15)-0.125, np.linspace(0, 1, 15)-0.5)
    coords = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)), axis=1)
    true_coords = np.einsum('ij,kj->ki',true_rot.numpy(), coords)
    pred_coords = np.einsum('ij,kj->ki',rotation_pred, coords)
    ax.cla()
    ax.scatter(true_coords[:,0], true_coords[:,1], true_coords[:,2], alpha=0.2, label='$p_{\gamma}^*$')
    ax.scatter(pred_coords[:,0], pred_coords[:,1], pred_coords[:,2], label='$p_{\gamma}$')
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.75,0.75)
    plt.axis('off')
    plt.legend()
    # plt.savefig(path + os.path.sep + f'{idx}.png', bbox_inches='tight')
    plt.close()
    # plt.draw()
    # plt.pause(0.0001)

def visualizeaffine16(logits, true, idx, path):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    true_rot = torch.cat((true[:,:3].reshape(1,3), true[:,4:7].reshape(1,3), true[:,8:11].reshape(1,3)),dim=0)
    rotation_pred = logits[:,:9].view(3,3).detach().cpu()
    u, s, vt = np.linalg.svd(rotation_pred.numpy())
    rotation_pred = u @ vt
    if np.linalg.det(rotation_pred) < 0:
        rotation_pred = -1 * rotation_pred
    X, Y, Z = np.meshgrid(np.linspace(0, 0.25, 15)-0.125, np.linspace(0, 0.25, 15)-0.125, np.linspace(0, 1, 15)-0.5)
    coords = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)), axis=1)
    true_coords = np.einsum('ij,kj->ki',true_rot.numpy(), coords)
    pred_coords = np.einsum('ij,kj->ki',rotation_pred, coords)
    ax.cla()
    ax.scatter(true_coords[:,0], true_coords[:,1], true_coords[:,2], alpha=0.2, label='$p_{\gamma}^*$')
    ax.scatter(pred_coords[:,0], pred_coords[:,1], pred_coords[:,2], label='$p_{\gamma}$')
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.75,0.75)
    plt.axis('off')
    plt.legend()
    # plt.savefig(path + os.path.sep + f'{idx}.png', bbox_inches='tight')
    plt.close()


def visualizequat7(logits, true, idx, path):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    logits = logits.detach().cpu()
    true_rot = torch.cat((true[:,:3].reshape(1,3), true[:,4:7].reshape(1,3), true[:,8:11].reshape(1,3)),dim=0)
    u_prime = logits[:,:3].view(1,3) # this is a impure quaternion
    phi = torch.linalg.norm(u_prime) 
    # print(phi)
    q = u_prime / phi
    theta = torch.sigmoid(logits[:,3]) * 2 * np.pi
    img = q * torch.sin(theta)
    # quaternion_pred = torch.tensor([torch.cos(theta),  img[:,0], img[:,1], img[:,2]]).to(device)
    rotation_pred = _quaternion_to_rotation(torch.cos(theta), img).detach().cpu()
    u, s, vt = np.linalg.svd(rotation_pred.numpy())
    rotation_pred = u @ vt
    if np.linalg.det(rotation_pred) < 0:
        rotation_pred = -1 * rotation_pred
    X, Y, Z = np.meshgrid(np.linspace(0, 0.25, 15)-0.125, np.linspace(0, 0.25, 15)-0.125, np.linspace(0, 1, 15)-0.5)
    coords = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)), axis=1)
    true_coords = np.einsum('ij,kj->ki',true_rot.numpy(), coords)
    pred_coords = np.einsum('ij,kj->ki',rotation_pred, coords)
    
    ax.cla()
    ax.scatter(true_coords[:,0], true_coords[:,1], true_coords[:,2], alpha=0.2)
    ax.scatter(pred_coords[:,0], pred_coords[:,1], pred_coords[:,2])
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.75,0.75)
    plt.axis('off')
    # plt.savefig(path + os.path.sep + f'{idx}.png', bbox_inches='tight')
    plt.close()
    # plt.draw()
    # plt.pause(0.0001)



def visualizeAngVel6(logits, true, idx, path):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    logits = logits.detach().cpu()
    identity = torch.eye(3)
    true_rot = torch.cat((true[:,:3].reshape(1,3), true[:,4:7].reshape(1,3), true[:,8:11].reshape(1,3)),dim=0)
    ang_vel = logits[:,:3].view(1,3)
    u = torch.linalg.norm(ang_vel)
    rotation_pred = identity + _skew(ang_vel / u).detach().cpu() * torch.sin(u) + _skew(ang_vel / u).detach().cpu()**2 * (1 - torch.cos(u))
    u, s, vt = np.linalg.svd(rotation_pred.numpy())
    rotation_pred = u @ vt
    if np.linalg.det(rotation_pred) < 0:
        rotation_pred = -1 * rotation_pred
    X, Y, Z = np.meshgrid(np.linspace(0, 0.25, 15)-0.125, np.linspace(0, 0.25, 15)-0.125, np.linspace(0, 1, 15)-0.5)
    coords = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)), axis=1)
    true_coords = np.einsum('ij,kj->ki',true_rot.numpy(), coords)
    pred_coords = np.einsum('ij,kj->ki',rotation_pred, coords)
    ax.cla()
    ax.scatter(true_coords[:,0], true_coords[:,1], true_coords[:,2], alpha=0.2)
    ax.scatter(pred_coords[:,0], pred_coords[:,1], pred_coords[:,2])
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.75,0.75)
    plt.axis('off')
    # plt.savefig(path + os.path.sep + f'{idx}.png', bbox_inches='tight')
    plt.close()
    # plt.draw()
    # plt.pause(0.0001)