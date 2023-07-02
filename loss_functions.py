# Defning the loss functions for different rotation parameterizations
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def _skew(vector):
    antisymmetric = torch.tensor([[0, -vector[:,2], vector[:,1]],
                                [vector[:,2], 0, -vector[:,0]],
                                [-vector[:,1], vector[:,0], 0]]).to(device)
    return antisymmetric

def _quaternion_to_rotation(real, img):
    q = (real, img[:,0], img[:,1], img[:,2])
    R = torch.tensor([[1 - (2*q[2]**2 - 2*q[3]**2), 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
            [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3]-2*q[1]*q[0]],
            [2*q[1]*q[3]-2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1-2*q[1]**2-2*q[2]**2]]).to(device)
    return R

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:

    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)




def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)




def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:

    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))



class LOSS_FUNCTION:
    def loss_RotMat12(logits, true, include_so3_constraints=False, gamma=10, alpha=3.0):
        identity = torch.eye(3).to(device)
        true_rot = torch.cat((true[:,:3].reshape(1,3), true[:,4:7].reshape(1,3), true[:,8:11].reshape(1,3)),dim=0).to(device)
        true_translation = torch.cat((true[:,3], true[:,7], true[:,11])).to(device)
        rotation_pred = logits[:,:9].view(3,3)
        u, _, vt = torch.svd(rotation_pred)
        rotation_pred = u @ vt
        if torch.linalg.det(rotation_pred) < 0:
            msk = torch.ones_like(u).to(device)
            msk[-1,:] = msk[-1,:]* (-1)
            rotation_pred = ((u*msk) @ vt)
        try:
            assert torch.abs(torch.linalg.det(rotation_pred) - 1) < 1e-3
        except:
            print('Invalid rotation, det is : ', torch.linalg.det(rotation_pred))
        rotation_loss = torch.linalg.norm(rotation_pred.T @ true_rot - identity) + torch.linalg.norm(rotation_pred @ true_rot.T - identity)
        translation_loss = torch.linalg.norm(logits[:,9:] - true_translation.reshape(1,3))
        if include_so3_constraints:
            so3_constraint_loss = torch.linalg.norm(rotation_pred.T - torch.inverse(rotation_pred)) + alpha * torch.abs(1 - torch.linalg.det(rotation_pred))
            loss = gamma * (rotation_loss) + translation_loss + so3_constraint_loss
        else:
            loss = gamma * (rotation_loss) + translation_loss 
        return loss

    def loss_AngVel6(logits, true, gamma=10.0):
        identity = torch.eye(3).to(device)
        true_rot = torch.cat((true[:,:3].reshape(1,3), true[:,4:7].reshape(1,3), true[:,8:11].reshape(1,3)),dim=0).to(device)
        print(torch.linalg.det(true_rot))
        true_translation = torch.cat((true[:,3], true[:,7], true[:,11])).to(device)
        u_prime = logits[:,:3].view(1,3)
        norm = (torch.linalg.norm(u_prime) + 1e-5)
        rotation_pred = identity + (_skew(u_prime / norm) * torch.sin(2*np.pi * torch.sigmoid(norm))) + \
                            (_skew(u_prime / norm)**2 * (1 - torch.cos(2*np.pi * torch.sigmoid(norm))))
        rotation_loss = torch.linalg.norm(rotation_pred.T @ true_rot - identity) + torch.linalg.norm(rotation_pred @ true_rot.T - identity)
        translation_loss = torch.linalg.norm(logits[:,3:] - true_translation.reshape(1,3))
        loss = gamma * (rotation_loss) + translation_loss 
        return loss
    
    def loss_quat7(logits, true, gamma=10.0):
        true_rot = torch.cat((true[:,:3].reshape(1,3), true[:,4:7].reshape(1,3), true[:,8:11].reshape(1,3)),dim=0).to(device)
        true_translation = torch.cat((true[:,3], true[:,7], true[:,11])).to(device)
        u_prime = logits[:,:3].view(1,3) # this is a impure quaternion
        phi = torch.linalg.norm(u_prime)
        q = u_prime / (phi + 1e-5)
        theta = logits[:,3]
        img = q * torch.sin(theta)
        rotation_pred = torch.tensor([torch.cos(theta), img[:,0], img[:,1], img[:,2]]).to(device)
        true_quat = matrix_to_quaternion(true_rot)
        rotation_loss = torch.linalg.norm(torch.flatten(quaternion_multiply(quaternion_invert(true_quat), rotation_pred)) - \
                torch.flatten(torch.tensor([1, 0, 0, 0]).to(device)))
        translation_loss = torch.linalg.norm(logits[:,4:] - true_translation.reshape(1,3))
        loss = gamma * (rotation_loss) + translation_loss 
        return loss
    
    def loss_affine16(logits, true, gamma=10.0):
        loss = gamma * torch.linalg.norm(logits - true)
        return loss
