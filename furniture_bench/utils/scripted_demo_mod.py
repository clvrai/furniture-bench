import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def scale_scripted_action(action: torch.Tensor, 
                          pos_bounds_m=0.01, 
                          ori_bounds_deg=3,
                          device='cpu') -> torch.Tensor:
    """Scale down the action that is provided from the scripted policy.
    Raw action comes in the form of a "delta EE pose", we can directly
    treat these as the position and rotation errors to be scaled down
    and used to create the modified action (which is then passed to the 
    DifferentialIK controller)

    Args:
        action (torch.Tensor): Raw action [pos, quat]

    Returns:
        torch.Tensor: Action with errors scaled down
    """
    position_error = action[0, :3]
    quat_error = action[0, 3:-1]
    gripper_act = action[:, -1]
    
    # scale down position error
    max_pos_value = torch.abs(position_error).max()
    clip_pos_value = pos_bounds_m
    if max_pos_value > clip_pos_value:
        pos_error_scalar = clip_pos_value / max_pos_value
        position_error = position_error * pos_error_scalar

    # convert rotation error to axis angle
    rotvec_error = R.from_quat(quat_error.cpu().numpy()).as_rotvec()
    if np.allclose(rotvec_error, 0.0):
        pass
    else:
        delta_norm = np.linalg.norm(rotvec_error)
        delta_axis = rotvec_error / delta_norm
        
        # scale down axis angle magnitude
        max_rot_radians = np.deg2rad(ori_bounds_deg)
        delta_norm_clipped = np.clip(delta_norm, a_min=0.0, a_max=max_rot_radians)
        delta_rotvec_scaled = delta_axis * delta_norm_clipped
        
        # convert back to quat, and create modified action
        quat_error = torch.from_numpy(R.from_rotvec(delta_rotvec_scaled).as_quat())
    action_scaled = torch.cat((position_error, quat_error, gripper_act), dim=-1).reshape(1, -1)

    return action_scaled.float().to(device)