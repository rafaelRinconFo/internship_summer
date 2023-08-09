import torch


def matrix_from_angles(rot):
    """Create a rotation matrix from a triplet of rotation angles.
    Args:
    rot: a torch.Tensor of shape [..., 3], where the last dimension is the rotation
        angles, along x, y, and z.
    Returns:
    A torch.tensor of shape [..., 3, 3], where the last two dimensions are the
    rotation matrix.
    This function mimics _euler2mat from struct2depth/project.py, for backward
    compatibility, but wraps tensorflow_graphics instead of reimplementing it.
    The negation and transposition are needed to bridge the differences between
    the two.
    """
    rank = rot.dim()
    # Swap the two last dimensions
    perm = torch.cat(
        [
            torch.arange(0, rank - 1, dtype=torch.long),
            torch.tensor([rank]),
            torch.tensor([rank - 1]),
        ],
        dim=0,
    )

    return euler_angles_to_matrix(-rot)


def angles_from_matrix(matrix):
    """Get a triplet of rotation angles from a rotation matrix.
    Args:
    matrix: A torch.tensor of shape [..., 3, 3], where the last two dimensions are
        the rotation matrix.
    Returns:
    A torch.Tensor of shape [..., 3], where the last dimension is the rotation
        angles, along x, y, and z.
    This function mimics _euler2mat from struct2depth/project.py, for backward
    compatibility, but wraps tensorflow_graphics instead of reimplementing it.
    The negation and transposition are needed to bridge the differences between
    the two.
    """
    rank = matrix.dim()
    # Swap the two last dimensions
    perm = torch.cat(
        [
            torch.arange(0, rank - 2, dtype=torch.long),
            torch.tensor([rank - 1]),
            torch.tensor([rank - 2]),
        ],
        dim=0,
    )
    return -matrix_to_euler_angles(matrix.permute(*perm), convention="XYZ")


def unstacked_matrix_from_angles(rx, ry, rz, name=None):
    """Create an unstacked rotation matrix from rotation angles.
    Args:
    rx: A torch.Tensor of rotation angles abound x, of any shape.
    ry: A torch.Tensor of rotation angles abound y (of the same shape as x).
    rz: A torch.Tensor of rotation angles abound z (of the same shape as x).
    name: A string, name for the op.
    Returns:
    A 3-tuple of 3-tuple of torch.Tensors of the same shape as x, representing the
    respective rotation matrix. The small 3x3 dimensions are unstacked into a
    tuple to avoid tensors with small dimensions, which bloat the TPU HBM
    memory. Unstacking is one of the recommended methods for resolving the
    problem.
    """
    angles = [-rx, -ry, -rz]
    sx, sy, sz = [torch.sin(a) for a in angles]
    cx, cy, cz = [torch.cos(a) for a in angles]
    m00 = cy * cz
    m10 = (sx * sy * cz) - (cx * sz)
    m20 = (cx * sy * cz) + (sx * sz)
    m01 = cy * sz
    m11 = (sx * sy * sz) + (cx * cz)
    m21 = (cx * sy * sz) - (sx * cz)
    m02 = -sy
    m12 = sx * cy
    m22 = cx * cy
    return ((m00, m01, m02), (m10, m11, m12), (m20, m21, m22))


def combine(rot_mat1, trans_vec1, rot_mat2, trans_vec2):
    """Composes two transformations, each has a rotation and a translation.
    Args:
    rot_mat1: A torch.tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec1: A torch.tensor of shape [..., 3] representing translation vectors.
    rot_mat2: A torch.tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec2: A torch.tensor of shape [..., 3] representing translation vectors.
    Returns:
    A tuple of 2 torch.Tensors, representing rotation matrices and translation
    vectors, of the same shapes as the input, representing the result of
    applying rot1, trans1, rot2, trans2, in succession.
    """
    # Building a 4D transform matrix from each rotation and translation, and
    # multiplying the two, we'd get:
    #
    # (  R2   t2) . (  R1   t1)  = (R2R1    R2t1 + t2)
    # (0 0 0  1 )   (0 0 0  1 )    (0 0 0       1    )
    #
    # Where each R is a 3x3 matrix, each t is a 3-long column vector, and 0 0 0 is
    # a row vector of 3 zeros. We see that the total rotation is R2*R1 and the t
    # total translation is R2*t1 + t2.
    r2r1 = torch.matmul(rot_mat2, rot_mat1)

    # Reshape the trans_vec1 tensor to have a compatible shape for matrix multiplication
    trans_vec1_reshaped = trans_vec1.view(
        trans_vec1.shape[0], trans_vec1.shape[1], -1
    )  # shape: (batch_size, num_channels, height * width)

    # Perform matrix multiplication
    result_reshaped = torch.matmul(rot_mat2, trans_vec1_reshaped)

    # Reshape the result back to the original shape of trans_vec1
    r2t1 = result_reshaped.view(trans_vec1.shape)
    return r2r1, r2t1 + trans_vec2


def euler_angles_to_matrix(euler_angles):
    batch_size = euler_angles.size(0)
    cos_theta = torch.cos(euler_angles[:, 0])
    sin_theta = torch.sin(euler_angles[:, 0])
    cos_phi = torch.cos(euler_angles[:, 1])
    sin_phi = torch.sin(euler_angles[:, 1])
    cos_psi = torch.cos(euler_angles[:, 2])
    sin_psi = torch.sin(euler_angles[:, 2])

    rotation_matrix = torch.zeros((batch_size, 3, 3), device=euler_angles.device)

    rotation_matrix[:, 0, 0] = (cos_theta * cos_psi)[:, 0, 0]
    rotation_matrix[:, 0, 1] = (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi)[
        :, 0, 0
    ]
    rotation_matrix[:, 0, 2] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)[
        :, 0, 0
    ]
    rotation_matrix[:, 1, 0] = (cos_theta * sin_psi)[:, 0, 0]
    rotation_matrix[:, 1, 1] = (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi)[
        :, 0, 0
    ]
    rotation_matrix[:, 1, 2] = (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi)[
        :, 0, 0
    ]
    rotation_matrix[:, 2, 0] = (-sin_theta)[:, 0, 0]
    rotation_matrix[:, 2, 1] = (sin_phi * cos_theta)[:, 0, 0]
    rotation_matrix[:, 2, 2] = (cos_phi * cos_theta)[:, 0, 0]

    return rotation_matrix


def matrix_to_euler_angles(rotation_matrix):
    batch_size = rotation_matrix.size(0)

    theta = torch.atan2(
        -rotation_matrix[:, 2, 0],
        torch.sqrt(rotation_matrix[:, 0, 0] ** 2 + rotation_matrix[:, 1, 0] ** 2),
    )
    phi = torch.atan2(rotation_matrix[:, 2, 1], rotation_matrix[:, 2, 2])
    psi = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])

    euler_angles = torch.stack((theta, phi, psi), dim=1)

    return euler_angles
