from plyfile import PlyData, PlyElement
import numpy as np
import torch

def write_gaussian_params_as_ply(means, scales, quats, opacities, ply_path):
    n_gaussians = means.shape[0]
    vertex = np.zeros(n_gaussians, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                        ('scale1', 'f4'), ('scale2', 'f4'), ('scale3', 'f4'),
                                        ('quat1', 'f4'), ('quat2', 'f4'), ('quat3', 'f4'), ('quat4', 'f4'),
                                        ('opacity', 'f4')])
    
    vertex['x'] = means[:, 0]
    vertex['y'] = means[:, 1]
    vertex['z'] = means[:, 2]
    vertex['scale1'] = scales[:, 0]
    vertex['scale2'] = scales[:, 1]
    vertex['scale3'] = scales[:, 2]
    vertex['quat1'] = quats[:, 0]
    vertex['quat2'] = quats[:, 1]
    vertex['quat3'] = quats[:, 2]
    vertex['quat4'] = quats[:, 3]
    vertex['opacity'] = opacities[:, 0]

    vertex_element = PlyElement.describe(vertex, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

def write_gaussian_params_as_visply(means, scales, quats, opacities, ply_path):
    n_gaussians = means.shape[0]
    # vertex = np.zeros(n_gaussians, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    #                                     ('scale1', 'f4'), ('scale2', 'f4'), ('scale3', 'f4'),
    #                                     ('quat1', 'f4'), ('quat2', 'f4'), ('quat3', 'f4'), ('quat4', 'f4'),
    #                                     ('opacity', 'f4')])
    
    
    
    
    # vertex['x'] = means[:, 0]
    # vertex['y'] = means[:, 1]
    # vertex['z'] = means[:, 2]
    # vertex['scale1'] = scales[:, 0]
    # vertex['scale2'] = scales[:, 1]
    # vertex['scale3'] = scales[:, 2]
    # vertex['quat1'] = quats[:, 0]
    # vertex['quat2'] = quats[:, 1]
    # vertex['quat3'] = quats[:, 2]
    # vertex['quat4'] = quats[:, 3]
    # vertex['opacity'] = opacities[:, 0]
    # vertex_element = PlyElement.describe(vertex, 'vertex')
    # ply_data = PlyData([vertex_element])
    # ply_data.write(ply_path)
    
    xyz = means
    normals = np.zeros_like(xyz)
    #f_dc size  n_gaussians*1*3   
    #f_dc 随机值 0-1
    f_dc = torch.rand((n_gaussians,1,3)).float()
    f_rest = torch.rand((n_gaussians,15,3)).float()
    # f_dc = torch.zeros((n_gaussians,1,3)).float()
    # f_rest = torch.zeros((n_gaussians,15,3)).float()
    # f_dc = np.zeros((n_gaussians,1,3))#不确定 0--final rgb  
    # f_rest = np.zeros((n_gaussians,15,3))

    f_dc = f_dc.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    f_rest = f_rest.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    opacities = opacities#应该是激活函数值
    scale = scales#应该是log值
    rotation = quats

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(n_gaussians, dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_path)


def read_gaussian_params_from_ply(ply_path):

    plydata = PlyData.read(ply_path)
    data = plydata['vertex']

    pos = np.hstack((data['x'][:, np.newaxis], data['y'][:, np.newaxis], data['z'][:, np.newaxis]))
    scales = np.hstack((data['scale1'][:, np.newaxis], data['scale2'][:, np.newaxis], data['scale3'][:, np.newaxis]))
    quats = np.hstack((data['quat1'][:, np.newaxis], data['quat2'][:, np.newaxis], data['quat3'][:, np.newaxis], data['quat4'][:, np.newaxis]))    
    opacities = data['opacity'][:, np.newaxis]

    return pos, scales, quats, opacities

def write_pts_with_major_dirs_as_ply(pos, dirs, ply_path):
    num_pts = pos.shape[0]
    vertex_with_dir = np.zeros(num_pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                           ('dir_x', 'f4'), ('dir_y', 'f4'), ('dir_z', 'f4')])
    
    vertex_with_dir['x'] = pos[:, 0]
    vertex_with_dir['y'] = pos[:, 1]
    vertex_with_dir['z'] = pos[:, 2]
    vertex_with_dir['dir_x'] = dirs[:, 0]
    vertex_with_dir['dir_y'] = dirs[:, 1]
    vertex_with_dir['dir_z'] = dirs[:, 2]

    vertex_element = PlyElement.describe(vertex_with_dir, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

def read_pts_with_major_dirs_from_ply(file_path):
    
    plydata = PlyData.read(file_path)
    data = plydata['vertex']

    pos = np.hstack((data['x'][:, np.newaxis], data['y'][:, np.newaxis], data['z'][:, np.newaxis]))
    dirs = np.hstack((data['dir_x'][:, np.newaxis], data['dir_y'][:, np.newaxis], data['dir_z'][:, np.newaxis]))

    return pos, dirs