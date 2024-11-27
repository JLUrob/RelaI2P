import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import math
import open3d as o3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
import torch

# function: 点云360°展开
# params: 点云
# return: 行坐标和列坐标
def point_cloud_to_panorama(points):
    # Extract x, y, z coordinates
    x = points[0, :]
    y = points[1, :]
    z = points[2, :]

    # Convert to polar coordinates (r, theta, z)
    r = np.sqrt(x**2 + y**2 + z**2).reshape(1,-1)
    theta_col = np.arctan2(y, x) * 180 / np.pi  # Convert to degrees  当前范围[-180，180]
    theta_row = np.arctan2(z , np.sqrt(x**2 + y**2)) * 180 / np.pi #当前范围大约[-30，5]
    # Normalize theta_col to [0, 360]  to [2.5, 37.5]->[0, 40]
    theta_col += 180
    theta_row += 32.5
    # Normalize theta_col to [0, 1440]  to [0, 160]
    theta_col = theta_col*4
    theta_row = theta_row*4
    # print(theta_col.max())
    # print(theta_col.min())
    # print(theta_row.max())
    # print(theta_row.min())
    
    #反转很重要
    theta_col  = 1440 - theta_col
    theta_row  = 160 - theta_row
    
    # theta_col  = (theta_col+720)%1440
    # theta_row  = (theta_row+80)%160
    
    # while(1):
    #     pass
    
    # theta_col += 36
    # theta_row += 12
    # print(theta_col.shape)
    # print(np.sum((theta_col<0)|(theta_col>360)))
    # print(theta_row.shape)
    # print(np.sum((theta_row<0)|(theta_row>120)))
    # while(1):
    #     pass
    # Sort points by theta 

    return theta_col, theta_row, r

# function: 360°图可视化
# params: 行，列，深度，点云
def plt_points(theta_col, theta_row, r, pc):
    # r, theta, z = point_cloud_to_panorama(xyz_points)
    # # Plot the panoramic view
    max_c = np.max(r)
    min_c = np.min(r)
    r = ((r - min_c) / (max_c - min_c)) * 255
    
    _pc = pc.T
    x_max_pc = np.max(_pc[:,0])
    x_min_pc = np.min(_pc[:,0])
    y_max_pc = np.max(_pc[:,1])
    y_min_pc = np.min(_pc[:,1])
    z_max_pc = np.max(_pc[:,2])
    z_min_pc = np.min(_pc[:,2])
    _pc[:,0] = ((_pc[:,0] - x_min_pc)/(x_max_pc - x_min_pc))*255
    _pc[:,1] = ((_pc[:,1] - y_min_pc)/(y_max_pc - y_min_pc))*255
    _pc[:,2] = ((_pc[:,2] - z_min_pc)/(z_max_pc - z_min_pc))*255
    
    img_map = np.zeros((160,1440,3))
    int_theta_col = theta_col.astype(int)%1440
    int_theta_row = theta_row.astype(int)%160
    int_r = r.astype(int)
    img_map[int_theta_row,int_theta_col,:] = _pc
    cv2.imwrite('360_xyz.jpg', img_map)
    
    img_map_r = np.zeros((160,1440))
    img_map_r[int_theta_row,int_theta_col] = int_r
    cv2.imwrite('360_r.jpg', img_map_r)
    # Display the file path to the saved image
    # output_file_path
    
    plt.figure(figsize=(8,4), dpi=600)
    # plt.figure(figsize=(72, 24))
    plt.imshow(img_map_r)
    plt.xticks(None)
    plt.yticks(None)
    plt.axis("off")
    plt.savefig('plot_img_map_r.jpg')

# functon: 获得pcd_360_img的原始输入图像
# theta_col: 行坐标  theta_row: 列坐标  pc：点云  r：深度  intensity: 反射强度  sn： 法向量
# return: pcd_360_img -->(160, 1440, 8)
def get_360_img(theta_col, theta_row, pc, r, intensity,sn):
    
    int_theta_col = theta_col.astype(int)%1440
    int_theta_row = theta_row.astype(int)%160
    pcd_360_img = np.zeros((160,1440,8))
    pcd_360_img[int_theta_row, int_theta_col,0:3] = pc.T
    pcd_360_img[int_theta_row, int_theta_col,3] = r
    pcd_360_img[int_theta_row, int_theta_col,4] = intensity
    pcd_360_img[int_theta_row, int_theta_col,5:8] = sn.T
    # print(pcd_360_img.shape)
    # print(pc.shape)
    # print(r.shape)
    # print(intensity.shape)
    # print(sn.shape)
    
    # for i in range(10):
    #     print("--------------------------------")
    #     print(pcd_360_img[int_theta_row[i], int_theta_col[i],0])
    #     print(pc[0,i])
    #     print(pcd_360_img[int_theta_row[i], int_theta_col[i],3])
    #     print(r[0,i])
    #     print(pcd_360_img[int_theta_row[i], int_theta_col[i],4])
    #     print(intensity[0,i])
    #     print(pcd_360_img[int_theta_row[i], int_theta_col[i],5])
    #     print(sn[0,i])
    
    # while(1):
    #     pass
    return pcd_360_img

# function: 生成高斯分布（numpy）
def generate_gaussian_distribution(center, size):
    """生成以center为中心的高斯分布,并归一化概率值"""
    # 生成高斯分布
    distribution = np.exp(-0.5 * (np.arange(size) - center)**2)
    # 归一化概率值
    distribution /= distribution.sum()
    return distribution

# function: 生成高斯分布（torch）
def tensor_generate_gaussian_distribution(center, size):
    """生成以center为中心的高斯分布,并归一化概率值"""
    # 生成高斯分布
    range_tensor = torch.arange(size, dtype=torch.float32).cuda()
    center_tensor = torch.tensor(center, dtype=torch.float32)
    distribution = torch.exp(-0.5 * (range_tensor - center_tensor)**2)
    # 归一化概率值
    distribution /= distribution.sum()
    return distribution

# function: 从calib.txt文件中读取内参，外参
class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(
            os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(
                self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape(
                        (3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]
                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4)
                        P[0:3, 3] = np.asarray([tx, ty, tz])
                        calib_matrix_dict[seq_int][key] = P
        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]

# class_function: 最远点采样（原论文和改进论文中没有用到此种采样数据）
class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int_)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx

def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]

# class_function: 制作kitti_dataset
class kitti_pc_img_dataset(data.Dataset):
    def __init__(self, root_path, mode, num_pc,
                 P_tx_amplitude=5, P_ty_amplitude=5, P_tz_amplitude=5,
                 P_Rx_amplitude=0, P_Ry_amplitude=2.0 * math.pi, P_Rz_amplitude=0,num_kpt=512,is_front=False):
        super(kitti_pc_img_dataset, self).__init__()
        self.root_path = root_path
        self.mode = mode
        self.dataset = self.make_kitti_dataset(root_path, mode)
        self.calibhelper = KittiCalibHelper(root_path)
        self.num_pc = num_pc
        self.img_H = 160
        self.img_W = 512

        self.P_tx_amplitude = P_tx_amplitude
        self.P_ty_amplitude = P_ty_amplitude
        self.P_tz_amplitude = P_tz_amplitude
        self.P_Rx_amplitude = P_Rx_amplitude
        self.P_Ry_amplitude = P_Ry_amplitude
        self.P_Rz_amplitude = P_Rz_amplitude
        self.num_kpt=num_kpt
        self.farthest_sampler = FarthestSampler(dim=3)

        self.node_a_num= 128
        self.node_b_num= 128
        self.is_front=is_front
        print('load data complete')
    
    # function: 制作每一对对应图像和点云的参数
    # params: root_path: kitti数据集的路径  mode：train/val
    # return: dataset（以列表的形式返回）
    def make_kitti_dataset(self, root_path, mode):
        dataset = []

        if mode == 'train':
            seq_list = list(range(9))
        elif 'val' == mode:
            seq_list = [9, 10]
        elif 'tiny_val' == mode:
            seq_list = [9]
        else:
            raise Exception('Invalid mode.')

        skip_start_end = 0
        for seq in seq_list:
            img2_folder = os.path.join(root_path, 'sequences', '%02d' % seq, 'image_2')
            img3_folder = os.path.join(root_path, 'sequences', '%02d' % seq, 'image_3')
            pc_folder = os.path.join(root_path, 'pre_data', '%02d' % seq, 'voxel0.1-SNr0.6')


            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder,
                                seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder,
                                seq, i, 'P3', sample_num))
        return dataset

    # function: 下采样
    # params: pointcloud: 点云数据
    #         intensity:  反射强度
    #         sn：        法向量
    #         voxel_grid_downsample_size: 下采样体素大小
    # return: pointcloud, intensity, sn（下采样之后的点云、反射强度、法向量）
    def downsample_with_intensity_sn(self, pointcloud, intensity, sn, voxel_grid_downsample_size):
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(np.transpose(pointcloud))
        intensity_max=np.max(intensity)

        fake_colors=np.zeros((pointcloud.shape[1],3))
        fake_colors[:,0:1]=np.transpose(intensity)/intensity_max

        pcd.colors=o3d.utility.Vector3dVector(fake_colors)
        pcd.normals=o3d.utility.Vector3dVector(np.transpose(sn))

        down_pcd=pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
        down_pcd_points=np.transpose(np.asarray(down_pcd.points))
        pointcloud=down_pcd_points

        intensity=np.transpose(np.asarray(down_pcd.colors)[:,0:1])*intensity_max
        sn=np.transpose(np.asarray(down_pcd.normals))

        return pointcloud, intensity, sn
    
    # function: 有些点云采样可能会不足self.num_pc（40960）个点，从下采样之后的点中补充至self.num_pc（40960）个点
    def downsample_np(self, pc_np, intensity_np, sn_np):
        if pc_np.shape[1] >= self.num_pc:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np=sn_np[:,choice_idx]
        return pc_np, intensity_np, sn_np

    # 本文没用过
    def search_for_accumulation(self, pc_folder, seq_pose_folder,
                                seq_i, seq_sample_num, Pc, P_oi,
                                stride):
        Pc_inv = np.linalg.inv(Pc)
        P_io = np.linalg.inv(P_oi)

        pc_np_list, intensity_np_list, sn_np_list = [], [], []

        counter = 0
        while len(pc_np_list) < 3:
            counter += 1
            seq_j = seq_i + stride * counter
            if seq_j < 0 or seq_j >= seq_sample_num:
                break

            npy_data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_j)).astype(np.float32)
            pc_np = npy_data[0:3, :]  # 3xN
            intensity_np = npy_data[3:4, :]  # 1xN
            sn_np = npy_data[4:7, :]  # 3xN

            # P_oj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['pose'].astype(np.float32)  # 4x4
            P_oj = np.load(os.path.join(seq_pose_folder, '%06d.npy' % seq_j))['pose'].astype(np.float32)  # 4x4
            P_ij = np.dot(P_io, P_oj)

            P_transform = np.dot(Pc_inv, np.dot(P_ij, Pc))
            pc_np = transform_pc_np(P_transform, pc_np)
            P_transform_rot = np.copy(P_transform)
            P_transform_rot[0:3, 3] = 0
            sn_np = transform_pc_np(P_transform_rot, sn_np)

            pc_np_list.append(pc_np)
            intensity_np_list.append(intensity_np)
            sn_np_list.append(sn_np)

        return pc_np_list, intensity_np_list, sn_np_list

    # function：读取点云数据
    # params： pc_folder: 点云数据
    #          seq_i:  序号
    # return: pc_np, intensity_np, sn_np（读取的点云数据）
    def get_pointcloud(self, pc_folder, seq_i):
        pc_path = os.path.join(pc_folder, '%06d.npy' % seq_i)
        npy_data = np.load(pc_path).astype(np.float32)
        # shuffle the point cloud data, this is necessary!
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = npy_data[3:4, :]  # 1xN
        sn_np = npy_data[4:7, :]  # 3xN

        return pc_np, intensity_np, sn_np

    # function: 内参修正
    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    # function: 内参尺度缩放
    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R


    def generate_random_transform(self):
        """
        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
             random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
             random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_folder, pc_folder, seq, seq_i, key, seq_sample_num = self.dataset[index]
        
        # ----------------------load image and points----------------
        img_path = os.path.join(img_folder, '%06d.png' % seq_i)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # HxWx3

        # load point cloud of seq_i
        P_Tr = np.dot(self.calibhelper.get_matrix(seq, key),
                      self.calibhelper.get_matrix(seq, 'Tr'))
        pc, intensity, sn = self.get_pointcloud(pc_folder, seq_i)
        # -------------------process pixels------------------------
        img = cv2.resize(img,
                         (int(round(img.shape[1] * 0.5)),
                          int(round((img.shape[0] * 0.5)))),
                         interpolation=cv2.INTER_LINEAR)
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]

        if 'train' == self.mode:
            img = self.augment_img(img)
        # -----------------get in_matrix --------------------------
        K = self.calibhelper.get_matrix(seq, key + '_K')
        K = self.camera_matrix_scaling(K, 0.5)
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)


        #1/4 scale
        K_4=self.camera_matrix_scaling(K,0.25)

        # -------------------------------------------------------
        
        # 360°展开 
        theta_col,theta_row,r = point_cloud_to_panorama(pc)
        # plt_points(theta_col, theta_row, r , pc)
        pcd_360_img = get_360_img(theta_col, theta_row, pc, r, intensity, sn)
        
        pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]
        sn = np.dot(P_Tr[0:3, 0:3], sn)
        
        # 原始点云重投影到像素上，是否在图像上，pc_mask保存点云能够映射到图像上标记为1
        ori_pc = np.dot(K_4, pc)
        ori_pc_mask = np.zeros((1, np.shape(ori_pc)[1]), dtype=np.float32)
        ori_pc[0:2, :] = ori_pc[0:2, :] / ori_pc[2:, :]
        
        #-----------选点方式-----------
        ori_xy = np.floor(ori_pc[0:2, :])     # 原文选点方式，保留整数部分
        # ori_xy = np.floor(ori_pc[0:2, :]+0.5)   # 修改后选点方式，就近选点
        
        ori_is_in_picture = (ori_xy[0, :] >= 0) & (ori_xy[0, :] <= (self.img_W*0.25 - 1)) & (ori_xy[1, :] >= 0) & (ori_xy[1, :] <= (self.img_H*0.25 - 1)) & (ori_pc[2, :] > 0)
        ori_pc_mask[:, ori_is_in_picture] = 1.
        
         # pc_kpt_idx最终保存的是随后，取出前num_kpt数量的映射在像素平面上点云点的索引(随机取)
        ori_pc_kpt_idx=np.where(ori_pc_mask.squeeze()==1)[0]
        
        # 计算平均y
        mean_y = np.mean(theta_col[ori_pc_kpt_idx])/4  # 波动比较小 采用mean
        region_selecttion = mean_y/30              # scale [0,12] 把360°按照30度划分区域                     
        # mid_y = np.median(theta_col[pc_kpt_idx])
        # print(mean_y)
        gaussian_distribution = generate_gaussian_distribution(center=region_selecttion, size=12)
        # print(gaussian_distribution.argmax())
        # print(gaussian_distribution.sum())
        # print(gaussian_distribution.shape)
        # print(mid_y)
        
        # -------------------process points------------------------
        pc, intensity, sn = self.downsample_with_intensity_sn(pc, intensity, sn, voxel_grid_downsample_size=0.1)

        pc, intensity, sn = self.downsample_np(pc, intensity,sn)

         # 点云重投影到像素上，是否在图像上，pc_mask保存点云能够映射到图像上标记为1
        pc_ = np.dot(K_4, pc)
        pc_mask = np.zeros((1, np.shape(pc)[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
        
        #-----------选点方式-----------
        xy = np.floor(pc_[0:2, :])     # 原文选点方式，保留整数部分
        # xy = np.floor(pc_[0:2, :]+0.5)   # 修改后选点方式，就近选点
        
        is_in_picture = (xy[0, :] >= 0) & (xy[0, :] <= (self.img_W*0.25 - 1)) & (xy[1, :] >= 0) & (xy[1, :] <= (self.img_H*0.25 - 1)) & (pc_[2, :] > 0)
        pc_mask[:, is_in_picture] = 1.
        
         # pc_kpt_idx最终保存的是随后，取出前num_kpt数量的映射在像素平面上点云点的索引(随机取)
        pc_kpt_idx=np.where(pc_mask.squeeze()==1)[0]
        index=np.random.permutation(len(pc_kpt_idx))[0:self.num_kpt]
        pc_kpt_idx=pc_kpt_idx[index]
        
         # pc_outline_idx最终保存的是取出前num_kpt数量的未映射在像素平面上点云点的索引(随机取)
        pc_outline_idx=np.where(pc_mask.squeeze()==0)[0]
        index=np.random.permutation(len(pc_outline_idx))[0:self.num_kpt]
        pc_outline_idx=pc_outline_idx[index]
        
         # 创建img_mask，用稀疏矩阵方法最终保存点云能够达到像素平面上的像素点的坐标（即为像素坐标）,img_mask是与原图像长款各位1/4大小的图像
         # src_img大小是（160，512）  img_mask大小是(40，128)，img_mask最终结果是点云能够打到的位置保存为1
        xy2 = xy[:, is_in_picture]
        img_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])), shape=(int(self.img_H*0.25), int(self.img_W*0.25))).toarray()
        img_mask = np.array(img_mask)
        img_mask[img_mask > 0] = 1.
        
         # 将能够映射到像素平面的上的points保存其对应的像素坐标，以一维形式保存
        img_kpt_index=xy[1,pc_kpt_idx]*self.img_W*0.25 +xy[0,pc_kpt_idx]
        # img_kpt_index=(xy[1,pc_kpt_idx]+0.5)*self.img_W*0.25 + (xy[0,pc_kpt_idx] + 0.5)
        # print('img_kpt_index:',img_kpt_index.shape)
        # print(img_kpt_index)
        
         # 保存img_mask中num_kpt个为零的点的像素坐标，以一维的形式保存
        img_outline_index=np.where(img_mask.squeeze().reshape(-1)==0)[0]
        index=np.random.permutation(len(img_outline_index))[0:self.num_kpt]
        img_outline_index=img_outline_index[index]

        P = self.generate_random_transform()

        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]

        sn = np.dot(P[0:3, 0:3], sn)
        
        # 最远点采样 这俩个参数后续并没有用到
        node_a_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice( pc.shape[1],
                                                                            self.node_a_num * 8,
                                                                            replace=False)],
                                                                            k=self.node_a_num)

        node_b_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice( pc.shape[1],
                                                                            self.node_b_num * 8,
                                                                            replace=False)],
                                                                            k=self.node_b_num)

        if pc_kpt_idx.shape[0] < 512:
            print(pc_kpt_idx.shape)
        if img_kpt_index.shape[0] < 512:
            print(img_kpt_index.shape)

        return {'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pcd_360_img':torch.from_numpy(pcd_360_img.astype(np.float32)).permute(2,0,1).contiguous(),
                'pc': torch.from_numpy(pc.astype(np.float32)),
                'intensity': torch.from_numpy(intensity.astype(np.float32)),
                'sn': torch.from_numpy(sn.astype(np.float32)),
                'K': torch.from_numpy(K_4.astype(np.float32)),
                'P': torch.from_numpy(np.linalg.inv(P).astype(np.float32)),
                'gt_distribution': torch.from_numpy(gaussian_distribution),

                'pc_mask': torch.from_numpy(pc_mask).float(),       #(1,40960)
                'img_mask': torch.from_numpy(img_mask).float(),     #(40,128)
                
                'pc_kpt_idx': torch.from_numpy(pc_kpt_idx),         #512
                'pc_outline_idx':torch.from_numpy(pc_outline_idx),  #512
                'img_kpt_idx':torch.from_numpy(img_kpt_index).long() ,      #512
                'img_outline_index':torch.from_numpy(img_outline_index).long(),
                'node_a':torch.from_numpy(node_a_np).float(),
                'node_b':torch.from_numpy(node_b_np).float()
                }
               
# function: 重投影
def projection_pc_img(pc_np, img, K, size=1):
    """

    :param pc_np: points in camera coordinate
    :param img: image of the same frame
    :param K: Intrinsic matrix
    :return:
    """
    img_vis = np.copy(img)
    H, W = img.shape[0], img.shape[1]

    pc_np_front = pc_np[:, pc_np[2, :]>1.0]  # 3xN

    pc_pixels = np.dot(K, pc_np_front)  # 3xN
    pc_pixels = pc_pixels / pc_pixels[2:, :]  # 3xN
    for i in range(pc_pixels.shape[1]):
        px = int(pc_pixels[0, i])
        py = int(pc_pixels[1, i])
        # determine a point on image plane
        if px>=0 and px<=W-1 and py>=0 and py<=H-1:
            cv2.circle(img_vis, (px, py), size, (255, 0, 0), -1)
    return img_vis

if __name__ == '__main__':
    dataset = kitti_pc_img_dataset('/home/jlurobot/wzy/Calibration/Lidar2Camera/VP2P-Match-main/data/kitti', 'val', 40960)
    data = dataset[1100]
    
    print(len(dataset))
    print(data['K'])
    print(data['pc'].size())
    print(data['img'].size())
    print(data['pcd_360_img'].size())
    print(data['pc_mask'].size())
    print(data['intensity'].size())
    print(data['node_a'].size())
    print(data['node_b'].size())
    print(data['pc_kpt_idx'].size())
    print(data['gt_distribution'].size())
    print(data['gt_distribution'])
    print(data['gt_distribution'].sum())
    # print(data['img'])
    img = np.array(data['img'].permute(1,2,0))*255
    cv2.imwrite('./img.jpg',img)
