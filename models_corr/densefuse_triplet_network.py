import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import cv2
import numpy as np
sys.path.append(os.getcwd())

from mymodels.pointTransformer.point_transformer import PointsEncoder_pointwise
from mymodels.network_img.resnet import Image_ResNet
from mymodels.spvnas.models.semantic_kitti.spvcnn import SPVCNN
# from mymodels.depthnet.labdepth import LDRN
# from mymodels.densefuse.net import DenseFuse_net
from mymodels.densefuse.net_globa import DenseFuse_net
from utils.options import Options

class VP2PMatchNet(nn.Module):
    def __init__(self,opt:Options, is_pc_norm=False):
        super(VP2PMatchNet, self).__init__()
        self.opt=opt
        self.point_transformer = PointsEncoder_pointwise(is_pc_norm)
        self.voxel_branch = SPVCNN(num_classes=128, cr=0.5, pres=0.05, vres=0.05)
        self.resnet = Image_ResNet()
        self.densefuse = DenseFuse_net(512,512)
        
        print('# point net initialized')
        print('# voxel net initialized')
        print('# pixel net initialized')
        print('# densefuse net initialized')
        
        self.pc_score_head=nn.Sequential(
            nn.Conv1d(128+512,128,1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,64,1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,1,1,bias=False),
            nn.Sigmoid())

        self.img_score_head=nn.Sequential(
            nn.Conv2d(64+512,128,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,1,bias=False),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,1,1,bias=False),
            nn.Sigmoid())

        self.img_feature_layer=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False))
        self.pc_feature_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,64,1,bias=False))


    def forward(self,pc,intensity,sn,img):
        input = torch.cat((pc, intensity, sn), dim=1)
        #--------------------------------depth---------------------------
        # _, img_depth = self.depthnet(img) #修改处
        # print(img_depth.size())
        # print(img.size())
        # img_depth = np.array(img_depth[0].permute(1,2,0).data.cpu())
        # cv2.imwrite('./img_depth.jpg',img_depth)
        # img = torch.cat((img, img_depth), 1)
        #----------------------------------------------------------------
        
        global_img_feat, pixel_wised_feat = self.resnet(img)
        global_pc_feat, point_wised_feat = self.point_transformer(input)
        
        input_voxel = torch.cat((pc, intensity, sn), dim=1).transpose(2,1).reshape(-1, 7)
        batch_inds = torch.arange(pc.shape[0]).reshape(-1,1).repeat(1,pc.shape[2]).reshape(-1, 1).cuda()
        corrds = pc.transpose(2,1) - torch.min(pc.transpose(2,1), dim=1, keepdim=True)[0]
        corrds = corrds.reshape(-1, 3)
        corrds = torch.round(corrds / 0.05)
        corrds = torch.cat((corrds, batch_inds), dim=-1)
        _, voxel_feat = self.voxel_branch(input_voxel, corrds, pc.shape[0])
        point_wised_feat = point_wised_feat + voxel_feat    # 融合点云和体素特征；
        
        #--------------------------------------------------------------------------------
        
        #-------------------------densefuse_处理global_feat------------------------------
        # print(global_img_feat.shape,global_pc_feat.shape)
        fusion_global_img_feat = self.densefuse.decoder(self.densefuse.encoder(global_img_feat.unsqueeze(-1).unsqueeze(-1))).squeeze()
        fusion_global_pc_feat = self.densefuse.decoder(self.densefuse.encoder(global_pc_feat.unsqueeze(-1).unsqueeze(-1))).squeeze()
        # print(fusion_global_img_feat.shape,fusion_global_pc_feat.shape)
        # 融合方法1_wzy
        global_img_feat = global_img_feat + fusion_global_img_feat
        global_pc_feat = global_pc_feat + fusion_global_pc_feat
        # 融合方法2_wzy
        # global_img_feat = fusion_global_img_feat
        # global_pc_feat = fusion_global_pc_feat
        #--------------------------------------------------------------------------------
        
        
        # 融合图像特征和点云特征；
        img_feat_fusion = torch.cat((pixel_wised_feat, global_pc_feat.unsqueeze(-1).unsqueeze(-1).repeat(1,1,pixel_wised_feat.shape[2],pixel_wised_feat.shape[3])), dim=1)
        pc_feat_fusion = torch.cat((point_wised_feat, global_img_feat.unsqueeze(-1).repeat(1,1,point_wised_feat.shape[2])), dim=1)

        # 分别计算图像和点云的得分；
        img_score = self.img_score_head(img_feat_fusion)
        pc_score = self.pc_score_head(pc_feat_fusion)
        
        #--------------------------------------------------------------------------------
        
        #---------------------------------用densefuse更改的特征处理-----------------------
        
        # 使用densefuse进行特征融合；
        # pixel_wised_feat = self.img_feature_layer(pixel_wised_feat)
        # point_wised_feat = self.pc_feature_layer(point_wised_feat)
        
        # en_pixel = self.densefuse.encoder(pixel_wised_feat)
        # de_pixel = self.densefuse.decoder(en_pixel)
        
        # en_pc = self.densefuse.encoder(point_wised_feat.unsqueeze(-2))
        # de_pc = self.densefuse.decoder(en_pc)
        
        # point_wised_feat=F.normalize(de_pc.squeeze(), dim=1,p=2)
        # pixel_wised_feat=F.normalize(de_pixel, dim=1,p=2)

        # return pixel_wised_feat ,point_wised_feat ,img_score,pc_score
        #--------------------------------------------------------------------------------
    
        #--------------------------------之前特征处理-------------------------------------
        
        pixel_wised_feat = self.img_feature_layer(pixel_wised_feat)
        point_wised_feat = self.pc_feature_layer(point_wised_feat)

        point_wised_feat=F.normalize(point_wised_feat, dim=1,p=2)
        pixel_wised_feat=F.normalize(pixel_wised_feat, dim=1,p=2)
        
        
        return pixel_wised_feat ,point_wised_feat ,img_score,pc_score
        #--------------------------------------------------------------------------------
        

if __name__=='__main__':
    opt=Options()
    pc=torch.rand(10,3,40960).cuda()
    intensity=torch.rand(10,1,40960).cuda()
    sn=torch.rand(10,3,40960).cuda()
    img=torch.rand(10,3,160,512).cuda()
    net=VP2PMatchNet(opt).cuda()
    a,b,c,d=net(pc,intensity,sn,img)
    print(a.size())
    print(b.size())
    print(c.size())
    print(d.size())
    print(torch.max(pc), torch.min(pc), torch.mean(pc))
    