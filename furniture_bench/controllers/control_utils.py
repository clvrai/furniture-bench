"""Code derived from https://github.com/StanfordVL/perls2 and https://github.com/ARISE-Initiative/robomimic

Utility functions for controlling the robot.
"""
import math

import torch


@torch.jit.script
def opspace_matrices(mass_matrix, J_full):
    """Compute the lambda and nullspace matrices for the operational space control."""
    # Optimize code above.
    lambda_full_inv = torch.matmul(J_full, torch.linalg.solve(mass_matrix, J_full.T))

    # take the inverses, but zero out small singular values for stability
    svd_u, svd_s, svd_v = torch.linalg.svd(lambda_full_inv)
    singularity_threshold = 0.05
    svd_s_inv = torch.tensor(
        [0.0 if x < singularity_threshold else float(1.0 / x) for x in svd_s]
    ).to(mass_matrix.device)
    lambda_full = svd_v.T.matmul(torch.diag(svd_s_inv)).matmul(svd_u.T)

    # nullspace
    Jbar = torch.linalg.solve(mass_matrix, J_full.t()).matmul(lambda_full)
    nullspace_matrix = torch.eye(J_full.shape[-1], J_full.shape[-1]).to(
        mass_matrix.device
    ) - torch.matmul(Jbar, J_full)

    return lambda_full, nullspace_matrix


@torch.jit.script
def sign(x: float, epsilon: float = 0.01):
    """Get the sign of a number"""
    if x > epsilon:
        return 1.0
    elif x < -epsilon:
        return -1.0
    return 0.0


@torch.jit.script
def nullspace_torques(
    mass_matrix: torch.Tensor,
    nullspace_matrix: torch.Tensor,
    initial_joint: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_kp: float = 10,
):
    """
    For a robot with redundant DOF(s), a nullspace exists which is orthogonal to the remainder of the controllable
    subspace of the robot's joints. Therefore, an additional secondary objective that does not impact the original
    controller objective may attempt to be maintained using these nullspace torques.
    This utility function specifically calculates nullspace torques that attempt to maintain a given robot joint
    positions @initial_joint with zero velocity using proportinal gain @joint_kp
    :Note: @mass_matrix, @nullspace_matrix, @joint_pos, and @joint_vel should reflect the robot's state at the current
    timestep
    Args:
        mass_matrix (torch.tensor): 2d array representing the mass matrix of the robot
        nullspace_matrix (torch.tensor): 2d array representing the nullspace matrix of the robot
        initial_joint (torch.tensor): Joint configuration to be used for calculating nullspace torques
        joint_pos (torch.tensor): Current joint positions
        joint_vel (torch.tensor): Current joint velocities
        joint_kp (float): Proportional control gain when calculating nullspace torques
    Returns:
          torch.tensor: nullspace torques
    """
    # kv calculated below corresponds to critical damping
    joint_kv = torch.sqrt(joint_kp) * 2
    # calculate desired torques based on gains and error
    pose_torques = torch.matmul(
        mass_matrix, (joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel)
    )
    # map desired torques to null subspace within joint torque actuator space
    nullspace_torques = torch.matmul(nullspace_matrix.t(), pose_torques)
    return nullspace_torques


@torch.jit.script
def cross_product(vec1, vec2):
    """Efficient cross product function"""
    mat = torch.tensor(
        (
            [0.0, float(-vec1[2]), float(vec1[1])],
            [float(vec1[2]), 0.0, float(-vec1[0])],
            [float(-vec1[1]), float(vec1[0]), 0.0],
        )
    ).to(vec1.device)
    return torch.matmul(mat, vec2)


@torch.jit.script
def orientation_error(desired, current):
    """Optimized function to determine orientation error from matrices"""

    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (
        cross_product(rc1, rd1) + cross_product(rc2, rd2) + cross_product(rc3, rd3)
    )
    return error


@torch.jit.script
def quat_conjugate(a):
    """Compute the conjugate of a quaternion"""
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def orientation_error_quat(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def set_goal_position(
    position_limit: torch.Tensor, set_pos: torch.Tensor
) -> torch.Tensor:
    """
    Calculates and returns the desired goal position, clipping the result accordingly to @position_limits.
    @set_pos must be specified to define a global goal position
    """
    # Clip goal position
    set_pos[0] = torch.clamp(set_pos[0], position_limit[0][0], position_limit[0][1])
    set_pos[1] = torch.clamp(set_pos[1], position_limit[1][0], position_limit[1][1])
    set_pos[2] = torch.clamp(set_pos[2], position_limit[2][0], position_limit[2][1])
    return set_pos


@torch.jit.script
def quat2mat(quaternion: torch.Tensor) -> torch.Tensor:
    """Converts given quaternion (x, y, z, w) to matrix.

    Args:
        quaternion: vec4 float angles
    Returns:
        3x3 rotation matrix
    """
    EPS = 1e-8
    inds = torch.tensor([3, 0, 1, 2])
    q = quaternion.clone().detach().float()[inds]

    n = torch.dot(q, q)
    if n < EPS:
        return torch.eye(3)
    q *= math.sqrt(2.0 / n)
    q2 = torch.outer(q, q)
    return torch.tensor(
        [
            [
                1.0 - float(q2[2, 2]) - float(q2[3, 3]),
                float(q2[1, 2]) - float(q2[3, 0]),
                float(q2[1, 3]) + float(q2[2, 0]),
            ],
            [
                float(q2[1, 2]) + float(q2[3, 0]),
                1.0 - float(q2[1, 1]) - float(q2[3, 3]),
                float(q2[2, 3]) - float(q2[1, 0]),
            ],
            [
                float(q2[1, 3]) - float(q2[2, 0]),
                float(q2[2, 3]) + float(q2[1, 0]),
                1.0 - float(q2[1, 1]) - float(q2[2, 2]),
            ],
        ]
    )


@torch.jit.script
def unit_vector(data: torch.Tensor):
    """Returns ndarray normalized by length, i.e. eucledian norm, along axis."""

    data = torch.clone(data)
    if data.ndim == 1:
        data /= math.sqrt(torch.dot(data, data))
        return data
    length = torch.atleast_1d(torch.sum(data * data))
    length = torch.sqrt(length)
    data /= length
    return data


@torch.jit.script
def quat_multiply(q1: torch.Tensor, q0: torch.Tensor):
    """Return multiplication of two quaternions.
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True
    """
    x0, y0, z0, w0 = float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])
    x1, y1, z1, w1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    return torch.tensor(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=torch.float32,
    )


@torch.jit.script
def quat_slerp(
    quat0: torch.Tensor,
    quat1: torch.Tensor,
    fraction: float,
    spin: float = 0,
    shortestpath: bool = True,
):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1.0, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True
    """
    EPS = 1e-8
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = torch.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    d = torch.clip(d, -1.0, 1.0)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


@torch.jit.script
def mat2quat(rmat: torch.Tensor):
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat: 3x3 rotation matrix
    Returns:
        vec4 float quaternion angles
    """
    M = rmat[:3, :3]

    m00 = float(M[0, 0])
    m01 = float(M[0, 1])
    m02 = float(M[0, 2])
    m10 = float(M[1, 0])
    m11 = float(M[1, 1])
    m12 = float(M[1, 2])
    m20 = float(M[2, 0])
    m21 = float(M[2, 1])
    m22 = float(M[2, 2])
    # symmetric matrix K
    K = torch.tensor(
        [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    ).to(rmat.device)
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = torch.linalg.eigh(K)
    inds = torch.tensor([3, 0, 1, 2])
    q1 = V[inds, torch.argmax(w)]
    if q1[0] < 0.0:
        q1 = -q1
    inds = torch.tensor([1, 2, 3, 0])
    return q1[inds]


@torch.jit.script
def mat2pose(hmat: torch.Tensor):
    """
    Converts a homogeneous 4x4 matrix into pose.

    Args:
        hmat: a 4x4 homogeneous matrix

    Returns:
        (pos, orn) tuple where pos is vec3 float in cartesian,
            orn is vec4 float quaternion
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


@torch.jit.script
def set_goal_orientation(set_ori: torch.Tensor):
    """
    Calculates and returns the desired goal orientation, clipping the result accordingly to @orientation_limits.
    @delta and @current_orientation must be specified if a relative goal is requested, else @set_ori must be
    an orientation matrix specified to define a global orientation
    If @axis_angle is set to True, then this assumes the input in axis angle form, that is,
        a scaled axis angle 3-array [ax, ay, az]
    """
    goal_orientation = quat2mat(set_ori)
    return goal_orientation


@torch.jit.script
def pose2mat(
    pos: torch.Tensor, quat: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Converts pose to homogeneous matrix.

    Args:
        pos: a (os, orn tuple where pos is vec3 float cartesian
        quat: orn is vec4 float quaternion.

    Returns:
        4x4 homogeneous matrix
    """
    homo_pose_mat = torch.zeros((4, 4)).to(device)
    homo_pose_mat[:3, :3] = quat2mat(quat)
    homo_pose_mat[:3, 3] = pos
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


@torch.jit.script
def to_homogeneous(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Givien position and rotation matrix, convert it into homogeneous matrix."""
    transform = torch.zeros((4, 4), device=pos.device)
    if pos.ndim == 2:
        transform[:3, 3:] = pos
    else:
        assert pos.ndim == 1
        transform[:3, 3] = pos
    transform[:3, :3] = rot
    transform[3, 3] = 1

    return transform
