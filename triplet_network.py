import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from models360.pointTransformer.point_transformer import PointsEncoder_pointwise
from models360.network_img.resnet import Image_ResNet
from models360.spvnas.models.semantic_kitti.spvcnn import SPVCNN
from models360.kitti_dataset import tensor_generate_gaussian_distribution, generate_gaussian_distribution
from utils.options import Options

# def generate_gaussian_distribution(center, size):
#     """生成以center为中心的高斯分布,并归一化概率值"""
#     # 生成高斯分布
#     distribution = np.exp(-0.5 * (np.arange(size) - center)**2)
#     # 归一化概率值
#     distribution /= distribution.sum()
#     return distribution

class VP2PMatchNet(nn.Module):
    def __init__(self,opt:Options, is_pc_norm=False):
        super(VP2PMatchNet, self).__init__()
        self.opt=opt
        self.point_transformer = PointsEncoder_pointwise(is_pc_norm)
        self.voxel_branch = SPVCNN(num_classes=128, cr=0.5, pres=0.05, vres=0.05)
        self.resnet = Image_ResNet('img')
        self.pcd_img_resnet = Image_ResNet('pcd_360_img')

        print('# point net initialized')
        print('# voxel net initialized')
        print('# pixel net initialized')
        print('# plxel net initialized')

        self.pc_score_head=nn.Sequential(
            nn.Conv1d(128+512,128,1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,64,1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,1,1,bias=False),
            nn.Sigmoid()
            )

        self.img_score_head=nn.Sequential(
            nn.Conv2d(64+512,128,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,1,bias=False),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,1,1,bias=False),
            nn.Sigmoid()
            )
        
        self.global_fusion_layer=nn.Sequential(
            nn.Linear(2048+512,1024,bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,360,bias=False),
            nn.ReLU(),
            nn.Linear(360,12,bias=False),
            nn.Sigmoid()
            )
        
        self.funsion_img_cloud_stage1=nn.Sequential(
            nn.Conv2d(64+256,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        
        self.funsion_img_cloud_stage2=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.Sigmoid())
        
        # self.img_feature_layer=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False))
        self.pc_feature_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,64,1,bias=False))


    def forward(self,pc,intensity,sn,img,pcd_360_img,gt_distribution):
    # def forward(self,pc,intensity,sn,img,pcd_360_img):
        input = torch.cat((pc, intensity, sn), dim=1)

        global_img_feat, pixel_wised_feat = self.resnet(img)
        global_pcd_360_feat, pcd_360_feat = self.pcd_img_resnet(pcd_360_img)
        
        # print("global_img_feat:",global_img_feat.shape)
        # print("pixel_wised_feat:",pixel_wised_feat.shape)
        # print("global_pcd_360_img:",global_pcd_360_feat.shape)
        # print("pcd_360_feat:",pcd_360_feat.shape)
        # while(1):
        #     pass
        funsion_gloal_feat = torch.cat((global_img_feat, global_pcd_360_feat),dim = 1) # (B,2048+512)
        col_feat = self.global_fusion_layer(funsion_gloal_feat) # (B,12)
        # print(col_feat.shape)
        
        # self.opt.model = 'test'
        
        # 监督注意力
        gt_mid_col = torch.tensor(gt_distribution.argmax(dim=1), dtype=torch.int32)
        pre_mid_col = torch.tensor(col_feat.argmax(dim=1), dtype=torch.int32)
        
        # 自学习注意力
        # mid_col = col_feat.argmax(dim=1)
        # pre_region_selection = (mid_col/30).astype(torch.int32)
        # pre_region_selection = torch.tensor((mid_col/30), dtype=torch.int32)
        pre_dist = torch.zeros((pre_mid_col.shape[0], 12)).cuda()
        for i in range(pre_mid_col.shape[0]):
            pre_dist[i,:] = tensor_generate_gaussian_distribution(pre_mid_col[i],12)
            # print(pre_dist[i])
            
        # 截取图像区域
        if self.opt.model == 'train':
            left_margin = gt_mid_col*30-64
            right_margin = gt_mid_col*30+64
        else: # test
            left_margin = pre_mid_col*30-64
            right_margin = pre_mid_col*30+64
        
        
        # 截取图像区域分三种情况
        selected_pcd_360_feat = torch.zeros((pixel_wised_feat.shape[0],pcd_360_feat.shape[1],pixel_wised_feat.shape[2],pixel_wised_feat.shape[3])).cuda()
        for i in range(pcd_360_feat.shape[0]):
            if left_margin[i] < 0:
                left_margin[i] +=360
                left_width = 360-left_margin[i]
                selected_pcd_360_feat[i,:,:,0:left_width]= pcd_360_feat[i,:,:,left_margin[i]:360]
                selected_pcd_360_feat[i,:,:,left_width:128]= pcd_360_feat[i,:,:,0:right_margin[i]]
            elif right_margin[i] > 360:
                left_width = 360-left_margin[i]
                right_margin[i] -= 360
                selected_pcd_360_feat[i,:,:,0:left_width]= pcd_360_feat[i,:,:,left_margin[i]:360]
                selected_pcd_360_feat[i,:,:,left_width:128]= pcd_360_feat[i,:,:,0:right_margin[i]]
            else:
                selected_pcd_360_feat[i,:,:,:] = pcd_360_feat[i,:,:,left_margin[i]:right_margin[i]]
    
        # print(selected_pcd_360_feat.shape)
        # print(mid_col.shape)
        # print(mid_col)
        fusion_feat = torch.cat((pixel_wised_feat, selected_pcd_360_feat), dim=1)
        # print(fusion_feat.shape)

        #_------------------------points-------------------------------------------------
        global_pc_feat, point_wised_feat = self.point_transformer(input)

        input_voxel = torch.cat((pc, intensity, sn), dim=1).transpose(2,1).reshape(-1, 7)
        batch_inds = torch.arange(pc.shape[0]).reshape(-1,1).repeat(1,pc.shape[2]).reshape(-1, 1).cuda()
        corrds = pc.transpose(2,1) - torch.min(pc.transpose(2,1), dim=1, keepdim=True)[0]
        corrds = corrds.reshape(-1, 3)
        corrds = torch.round(corrds / 0.05)
        corrds = torch.cat((corrds, batch_inds), dim=-1)
        _, voxel_feat = self.voxel_branch(input_voxel, corrds, pc.shape[0])
        point_wised_feat = point_wised_feat + voxel_feat
        
        #--------------------------------------------------------------------------------
        
        fusion_feat = self.funsion_img_cloud_stage1(fusion_feat)
        
        fusion_feat_wise = self.funsion_img_cloud_stage2(fusion_feat)
        
        fusion_feat = fusion_feat * fusion_feat_wise
        
        #--------------------------------------------------------------------------------
        
        img_feat_fusion = torch.cat((pixel_wised_feat, global_pc_feat.unsqueeze(-1).unsqueeze(-1).repeat(1,1,pixel_wised_feat.shape[2],pixel_wised_feat.shape[3])), dim=1)
        pc_feat_fusion = torch.cat((point_wised_feat, global_img_feat.unsqueeze(-1).repeat(1,1,point_wised_feat.shape[2])), dim=1)

        img_score = self.img_score_head(img_feat_fusion)
        pc_score = self.pc_score_head(pc_feat_fusion)
        
        #--------------------------------------------------------------------------------
        
        # pixel_wised_feat = self.img_feature_layer(pixel_wised_feat)
        pixel_wised_feat = fusion_feat
        point_wised_feat = self.pc_feature_layer(point_wised_feat)
        point_wised_feat=F.normalize(point_wised_feat, dim=1,p=2)
        pixel_wised_feat=F.normalize(pixel_wised_feat, dim=1,p=2)


        
        return pixel_wised_feat, point_wised_feat , img_score, pc_score, pre_dist

if __name__=='__main__':
    opt=Options()
    pc=torch.rand(4,3,40960).cuda()
    intensity=torch.rand(4,1,40960).cuda()
    sn=torch.rand(4,3,40960).cuda()
    img=torch.rand(4,3,160,512).cuda()
    pcd_360_img=torch.rand(4,8,160,1440).cuda()
    
    gt_distribution = np.zeros((4,12))
    gt_distribution[0,:] = generate_gaussian_distribution(6,12)
    gt_distribution[1,:] = generate_gaussian_distribution(6,12)
    gt_distribution[2,:] = generate_gaussian_distribution(6,12)
    gt_distribution[3,:] = generate_gaussian_distribution(6,12)
    gt_distribution = torch.from_numpy(gt_distribution).cuda()
    net=VP2PMatchNet(opt).cuda()
    a,b,c,d,e=net(pc,intensity,sn,img,pcd_360_img,gt_distribution)
    # a,b,c,d,e=net(pc,intensity,sn,img,pcd_360_img)
    print(a.size())
    print(b.size())
    print(c.size())
    print(d.size())
    print(e.size())
    # print(e.device)
    print(torch.max(pc), torch.min(pc), torch.mean(pc))
    