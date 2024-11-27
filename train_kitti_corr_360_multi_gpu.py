import os
import torch
import argparse
# from models.pointTransformer.network_transformer import TransformerI2P_Voxel
from models_corr_360.triplet_network import VP2PMatchNet
# from mymodels.quadruplet_network import VP2PMatchNet
from models_corr_360.kitti_dataset import kitti_pc_img_dataset
# import loss
import utils.loss as loss
from utils.corr import ContrastiveCorrelationLoss
import numpy as np
import datetime
import logging
import math
# import options
import utils.options as options
import cv2
from scipy.spatial.transform import Rotation
# from utils.tools import angles2rotation_matrix_torch, matrix_to_quaternion, rotationErrors, translationErrors
from utils.tools import matrix_to_quaternion

from utils.epropnp.epropnp import EProPnP6DoF
from utils.epropnp.levenberg_marquardt import LMSolver, RSLMSolver
from utils.epropnp.camera import PerspectiveCamera
from utils.epropnp.cost_fun import AdaptiveHuberPnPCost
import torch.nn as nn
from collections import OrderedDict

class MonteCarloPoseLoss(nn.Module):

    def __init__(self, init_norm_factor=1.0, momentum=0.1):
        super(MonteCarloPoseLoss, self).__init__()
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
        """
        if self.training:
            with torch.no_grad():
                self.norm_factor.mul_(
                    1 - self.momentum).add_(self.momentum * norm_factor)

        loss_tgt = cost_target
        loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

        loss_pose = loss_tgt + loss_pred  # (num_obj, )
        loss_pose[torch.isnan(loss_pose)] = 0
        loss_pose = loss_pose.mean() / self.norm_factor

        return loss_pose.mean()

class Epro_PnP(nn.Module):

    def __init__(
            self,
            num_points=512,  # number of 2D-3D pairs
            mlp_layers=[1024],  # a single hidden layer
            epropnp=EProPnP6DoF(
                mc_samples=512,
                num_iter=4,
                solver=LMSolver(
                    dof=6,
                    num_iter=10,
                    init_solver=RSLMSolver(
                        dof=6,
                        num_points=8,
                        num_proposals=128,
                        num_iter=5))),
            camera=PerspectiveCamera(),
            cost_fun=AdaptiveHuberPnPCost(
                relative_delta=0.5)):
        super().__init__()
        self.num_points = num_points
        mlp_layers = [7] + mlp_layers
        mlp = []
        for i in range(len(mlp_layers) - 1):
            mlp.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            mlp.append(nn.LeakyReLU())
        mlp.append(nn.Linear(mlp_layers[-1], num_points * (3 + 2 + 2)))
        self.mlp = nn.Sequential(*mlp)
        # Here we use static weight_scale because the data noise is homoscedastic
        self.log_weight_scale = nn.Parameter(torch.zeros(2))
        self.epropnp = epropnp
        self.camera = camera
        self.cost_fun = cost_fun

    def forward_train(self, x3d, x2d, w2d, cam_mats, out_pose):

        w2d = (w2d.log_softmax(dim=-2) + self.log_weight_scale).exp()
        self.camera.set_param(cam_mats)
        self.cost_fun.set_param(x2d.detach(), w2d)  # compute dynamic delta
        pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = self.epropnp.monte_carlo_forward(
            x3d,
            x2d,
            w2d,
            self.camera,
            self.cost_fun,
            pose_init=out_pose,
            force_init_solve=True,
            with_pose_opt_plus=True)  # True for derivative regularization loss
        norm_factor = self.log_weight_scale.detach().exp().mean()
        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt, norm_factor

    def forward_test(self, in_pose, cam_mats, fast_mode=False):
        x3d, x2d, w2d = self.forward_correspondence(in_pose)
        self.camera.set_param(cam_mats)
        self.cost_fun.set_param(x2d.detach(), w2d)
        # returns a mode of the distribution
        pose_opt, _, _, _ = self.epropnp(
            x3d, x2d, w2d, self.camera, self.cost_fun,
            fast_mode=fast_mode)  # fast_mode=True activates Gauss-Newton solver (no trust region)
        return pose_opt

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff

def get_correspondence(pc_all, K_all,img_feature_all,pc_feature_all,img_score_all,pc_score_all):
    corr_3D = []
    corr_2D = []
    sel_num_img = 1000
    sel_num_pc = 3000
    for i in range(img_feature_all.shape[0]):
        pc = pc_all[i]
        img_feature = img_feature_all[i]
        pc_feature = pc_feature_all[i]
        img_score = img_score_all[i]
        pc_score = pc_score_all[i]
        K = K_all[i]

        img_x=np.linspace(0,np.shape(img_feature)[-1]-1,np.shape(img_feature)[-1]).reshape(1,-1).repeat(np.shape(img_feature)[-2],0).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
        img_y=np.linspace(0,np.shape(img_feature)[-2]-1,np.shape(img_feature)[-2]).reshape(-1,1).repeat(np.shape(img_feature)[-1],1).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])

        img_xy=np.concatenate((img_x,img_y),axis=0)
        img_xy = torch.tensor(img_xy).to(args.device)

        img_xy_flatten=img_xy.reshape(2,-1)
        img_feature_flatten=img_feature.reshape(np.shape(img_feature)[0],-1)
        img_score_flatten=img_score.squeeze().reshape(-1)
        
        # -----------------------------------------------------------------------
        img_sort_indices = torch.argsort(-img_score_flatten)[:sel_num_img]
        # if img_score_flatten[torch.argsort(-img_score_flatten)[sel_num_img]] < 0.9:
        if img_score_flatten[torch.argsort(-img_score_flatten)[sel_num_img]] < 0.9 and img_score_flatten[torch.argsort(-img_score_flatten)[0]] > 0.9:
            img_sort_indices = img_score_flatten > 0.9
            
        img_xy_flatten_sel = img_xy_flatten[:,img_sort_indices]
        img_feature_flatten_sel=img_feature_flatten[:,img_sort_indices]

        pc_sort_indices = torch.argsort(-pc_score[0])[:sel_num_pc]
        # if pc_score[0][torch.argsort(-pc_score[0])[sel_num_pc]] < 0.9:
        if pc_score[0][torch.argsort(-pc_score[0])[sel_num_pc]] < 0.9 and pc_score[0][torch.argsort(-pc_score[0])[0]] > 0.9:
            pc_sort_indices = pc_score[0] > 0.9

        pc_sel=pc[:,pc_sort_indices]
        pc_feature_sel=pc_feature[:,pc_sort_indices]
        # -----------------------------------------------------------------------

        dist=1-torch.sum(img_feature_flatten_sel.unsqueeze(2)*pc_feature_sel.unsqueeze(1), dim=0)

        sel_index=torch.argsort(dist,dim=1)[:,0]
        pc_sel=pc_sel[:,sel_index]
        img_xy_pc=img_xy_flatten_sel

        # -----------------------------------------------------------------------

        if img_xy_pc.shape[1] < sel_num_img:
            rand_ind = torch.randint(img_xy_pc.shape[1], (sel_num_img-img_xy_pc.shape[1],))
            img_xy_pc = torch.cat((img_xy_pc, img_xy_pc[:,rand_ind]),dim=1)
            pc_sel = torch.cat((pc_sel, pc_sel[:,rand_ind]),dim=1)

        corr_2D.append(img_xy_pc.transpose(1,0).unsqueeze(0)) # 
        corr_3D.append(pc_sel.transpose(1,0).unsqueeze(0)) # 

    return torch.cat(corr_2D,dim=0), torch.cat(corr_3D,dim=0) # corr_2D, corr_3D # 

def get_parse():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--epoch', type=int, default=35, metavar='epoch',
                        help='number of epoch to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='epoch',
                        help='start number of epoch to train')
    parser.add_argument('--train_batch_size', type=int, default=3, metavar='train_batch_size',
                        help='Size of train batch')
    parser.add_argument('--val_batch_size', type=int, default=2, metavar='val_batch_size',
                        help='Size of val batch')
    parser.add_argument('--data_path', type=str, default='./data/kitti/', metavar='data_path',
                        help='train and test data path')
    parser.add_argument('--num_workers', type=int, default=8, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--val_freq', type=int, default=1000, metavar='val_freq',
                        help='')
    parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
                        help='')
    parser.add_argument('--min_lr', type=float, default=0.00001, metavar='lr',
                        help='')

    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=2*math.pi*0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2*math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=2*math.pi*0, metavar='P_Rz_amplitude',
                        help='')

    parser.add_argument('--save_path', type=str, default='./res', metavar='save_path',
                        help='path to save log and model')
    parser.add_argument('--exp_name', type=str, default='./log', metavar='save_path',
                    help='path to save log and model')

    parser.add_argument('--num_kpt', type=int, default=512, metavar='num_kpt',
                        help='')
    parser.add_argument('--dist_thres', type=float, default=1, metavar='num_kpt',    # save radius
                        help='')
    # parser.add_argument('--dist_thres', type=float, default=0.5, metavar='num_kpt',    # save radius
    #                     help='')     

    parser.add_argument('--img_thres', type=float, default=0.95, metavar='img_thres',
                        help='')
    parser.add_argument('--pc_thres', type=float, default=0.95, metavar='pc_thres',
                        help='')

    parser.add_argument('--pos_margin', type=float, default=0.2, metavar='pos_margin',
                        help='')
    parser.add_argument('--neg_margin', type=float, default=1.8, metavar='neg_margin',
                        help='')

    parser.add_argument('--input_pt_num', type=int, default=40960, metavar='num_workers',
                        help='num of CPUs')

    parser.add_argument('--update_lr', type=int, default=20, metavar='update_lr',
                        help='update_lr')
                        
    parser.add_argument('--load_ckpt', type=str, default='none', metavar='save_path',
                    help='path to save log and model')

    parser.add_argument('--rank', type=int, default=0, metavar='rank',
                        help='id of gpu of current node')
    
    parser.add_argument('--local_rank', type=int, default=0, metavar='local_rank',
                        help='id of gpu of current node')
    
    parser.add_argument('--world_size', type=int, default=4, metavar='world_size',
                        help='number of gpu')
    
    parser.add_argument('--device', type=str, default='cuda', metavar='device',
                        help='use type of device')
    
    args = parser.parse_args()
    
    return args

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
    else:
        print("Not using  distributed mode")
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier() # 等待所有gpu运行到这

if __name__=='__main__':
    
    args = get_parse()
    init_distributed_mode(args)
    
    if torch.cuda.is_available():
        print("Cuda is available!")
        if torch.cuda.device_count() > 1 and args.local_rank == 0:
            print("Find {} gpus!".format(torch.cuda.device_count()))
        if torch.cuda.device_count() <= 1 :
            print("Too few gpu!")
            exit()
    else:
        print("Cuda is not available!")
        exit()
        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logdir=os.path.join(args.save_path, args.exp_name)
    try:
        os.makedirs(logdir)
    except:
        if args.rank == 0:
            print('mkdir failue')

    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (logdir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    opt=options.Options()
    opt.input_pt_num = args.input_pt_num
    
    train_dataset = kitti_pc_img_dataset(args.data_path, 'train', opt.input_pt_num,
                                         P_tx_amplitude=args.P_tx_amplitude,
                                         P_ty_amplitude=args.P_ty_amplitude,
                                         P_tz_amplitude=args.P_tz_amplitude,
                                         P_Rx_amplitude=args.P_Rx_amplitude,
                                         P_Ry_amplitude=args.P_Ry_amplitude,
                                         P_Rz_amplitude=args.P_Rz_amplitude,num_kpt=args.num_kpt,is_front=False)
    test_dataset = kitti_pc_img_dataset(args.data_path, 'val', opt.input_pt_num,
                                        P_tx_amplitude=args.P_tx_amplitude,
                                        P_ty_amplitude=args.P_ty_amplitude,
                                        P_tz_amplitude=args.P_tz_amplitude,
                                        P_Rx_amplitude=args.P_Rx_amplitude,
                                        P_Ry_amplitude=args.P_Ry_amplitude,
                                        P_Rz_amplitude=args.P_Rz_amplitude,num_kpt=args.num_kpt,is_front=False)
    assert len(train_dataset) > 10
    assert len(test_dataset) > 10
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    trainloader=torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.train_batch_size,
                                            sampler=train_sampler,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=args.num_workers)
    testloader=torch.utils.data.DataLoader(test_dataset,
                                           batch_size=args.val_batch_size,
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=args.num_workers)
    
    # model=TransformerI2P_Voxel(opt).to(args.device)
    model=VP2PMatchNet(opt).to(args.device)
    
    if args.local_rank == 0:
        try:
            os.remove(os.path.join(logdir,'kitti_init_mode_cuda0.t7')) 
        except:
            print('There is no init_model of cuda0, and start save the init mode of cuda0.')
        if args.load_ckpt != 'none':
            if args.local_rank == 0:
                print('start load pretrained model.')
            state_dict =torch.load(os.path.join(logdir,args.load_ckpt))
            # state_dict = OrderedDict((key[7:], value) for key, value in state_dict.items()) # 去除前缀'moudle.'
            model.load_state_dict(state_dict)
            args.start_epoch = int(args.load_ckpt.split('_')[-1].split('.')[0]) + 1
            
        torch.save(model.state_dict(), os.path.join(logdir,'kitti_init_mode_cuda0.t7'))
    torch.distributed.barrier() # 同步GPU
    model.load_state_dict(torch.load(os.path.join(logdir,'kitti_init_mode_cuda0.t7'), map_location=torch.device(args.device, index=int(args.local_rank))))
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.local_rank) # 同步BN
    model= torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    # model= torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
    if args.local_rank == 0:
        print('Successfully deployed the distribute model!')
    
    mc_loss_fun = MonteCarloPoseLoss().to(args.device)
    epro_pnp = Epro_PnP().to(args.device)

    warn_up = len(trainloader) * 2
    max_iter = len(trainloader) * args.epoch

    lambda1 = lambda iter_step: (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)

    start_epoch = args.start_epoch

    current_lr=args.lr
    learnable_params=filter(lambda p:p.requires_grad,model.parameters())
    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)

    global_step=start_epoch * len(trainloader)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler.last_epoch = global_step
    corr_loss_method = ContrastiveCorrelationLoss().to(args.device)
    
    logger.info(args)

    best_t_diff=1000
    best_r_diff=1000

    for epoch in range(start_epoch, args.epoch):
        for step,data in enumerate(trainloader):
            torch.cuda.empty_cache()
            global_step+=1
            model.train()
            optimizer.zero_grad()
            img=data['img'].to(args.device)                  #full size
            pc=data['pc'].to(args.device)
            intensity=data['intensity'].to(args.device)
            sn=data['sn'].to(args.device)
            K=data['K'].to(args.device)
            P=data['P'].to(args.device)
            pc_mask=data['pc_mask'].to(args.device)      
            img_mask=data['img_mask'].to(args.device)        #1/4 size
            B=img_mask.size(0)
            pc_kpt_idx=data['pc_kpt_idx'].to(args.device)    #(B,512)
            pc_outline_idx=data['pc_outline_idx'].to(args.device)
            img_kpt_idx=data['img_kpt_idx'].to(args.device)
            img_outline_idx=data['img_outline_index'].to(args.device)
            node_a=data['node_a'].to(args.device)
            node_b=data['node_b'].to(args.device)
            gt_dist = data['gt_distribution'].cuda()
            pcd_360_img=data['pcd_360_img'].cuda()
            img_x=torch.linspace(0,img_mask.size(-1)-1,img_mask.size(-1)).view(1,-1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).to(args.device)
            img_y=torch.linspace(0,img_mask.size(-2)-1,img_mask.size(-2)).view(-1,1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).to(args.device)
            img_xy=torch.cat((img_x,img_y),dim=1)
            
            rot_mat = P[:, :3, :3]  # (bs, 3, 3)
            trans_vec = P[:, :3, 3]  # (bs, 3)
            rot_quat = matrix_to_quaternion(rot_mat)
            pose_gt = torch.cat((trans_vec, rot_quat), dim=-1)  # B, 7

            
            img_features,pc_features,img_score,pc_score,pre_dist,feat_map=model(pc,intensity,sn,img,pcd_360_img,gt_dist)    #64 channels feature

            if epoch > -1:
                corr_2D, corr_3D = get_correspondence(pc, K, img_features,pc_features,img_score,pc_score)

                x2d = corr_2D.float()   # B, N, 2
                x3d = corr_3D   # B, N, 3
                w2d = torch.ones_like(x2d) # B, N, 2 .float().to(args.device)
                
                batch_cam_mats = K # B, 3, 3
                _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt, norm_factor = epro_pnp.forward_train(
                    x3d, x2d, w2d,
                    batch_cam_mats,
                    pose_gt)

                loss_mc = mc_loss_fun(
                    pose_sample_logweights, cost_tgt, norm_factor)

                loss_t = (pose_opt_plus[:, :3] - pose_gt[:, :3]).norm(dim=-1)
                beta = 0.05
                loss_t = torch.where(loss_t < beta, 0.5 * loss_t.square() / beta,
                                    loss_t - 0.5 * beta)
                loss_t = loss_t.mean()

                dot_quat = (pose_opt_plus[:, None, 3:] @ pose_gt[:, 3:, None]).squeeze(-1).squeeze(-1)
                loss_r = (1 - dot_quat.square()) * 2
                loss_r = loss_r.mean()

                loss_epro = loss_mc
                loss_reg = loss_t + loss_r
            
            # pc_kpt_idx是能映射到图像的点云的索引
            # pc_outline_idx是不能映射到图像的点云的索引
            pc_features_inline=torch.gather(pc_features,index=pc_kpt_idx.unsqueeze(1).expand(B,pc_features.size(1),args.num_kpt),dim=-1)
            pc_features_outline=torch.gather(pc_features,index=pc_outline_idx.unsqueeze(1).expand(B,pc_features.size(1),args.num_kpt),dim=-1)
            pc_xyz_inline=torch.gather(pc,index=pc_kpt_idx.unsqueeze(1).expand(B,3,args.num_kpt),dim=-1)
            pc_score_inline=torch.gather(pc_score,index=pc_kpt_idx.unsqueeze(1),dim=-1)
            pc_score_outline=torch.gather(pc_score,index=pc_outline_idx.unsqueeze(1),dim=-1)
            
            
            img_features_flatten=img_features.contiguous().view(img_features.size(0),img_features.size(1),-1)
            img_score_flatten=img_score.contiguous().view(img_score.size(0),img_score.size(1),-1)
            img_xy_flatten=img_xy.contiguous().view(img_features.size(0),2,-1)
            img_features_flatten_inline=torch.gather(img_features_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),args.num_kpt),dim=-1)
            img_xy_flatten_inline=torch.gather(img_xy_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,2,args.num_kpt),dim=-1)
            img_score_flatten_inline=torch.gather(img_score_flatten,index=img_kpt_idx.unsqueeze(1),dim=-1)
            img_features_flatten_outline=torch.gather(img_features_flatten,index=img_outline_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),args.num_kpt),dim=-1)
            img_score_flatten_outline=torch.gather(img_score_flatten,index=img_outline_idx.unsqueeze(1),dim=-1)
            


            pc_xyz_projection=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc_xyz_inline)+P[:,0:3,3:]))
            pc_xy_projection=pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]
            
            # 保存inline的像素点和重投影后的点云坐标点距离小于dist_thres（r）的索引，correspondence_mask最终为batch_size*512*512大小的图，小于dist_thres的标记为1，大于为0
            correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=1))<=args.dist_thres).float()   

            # loss_desc,dists=loss.circle_loss(img_features_flatten_inline,pc_features_inline,correspondence_mask,margin=0.25)
            loss_desc,dists=loss.adapt_loss(img_features_flatten_inline,pc_features_inline,correspondence_mask,margin=0.25)

            # loss_det=loss.det_loss2(img_score_flatten_inline.squeeze(),img_score_flatten_outline.squeeze(),pc_score_inline.squeeze(),pc_score_outline.squeeze(),dists,correspondence_mask)
            loss_det=loss.det_loss(img_score_flatten_inline.squeeze(),img_score_flatten_outline.squeeze(),pc_score_inline.squeeze(),pc_score_outline.squeeze(),dists,correspondence_mask)
            
            loss_corr = corr_loss_method(feat_map["pixel_feat_0"], feat_map["pcd_360_feat_0"], feat_map["pixel_feat_1"], feat_map["pcd_360_feat_1"])
            # loss_dist = loss.dist_loss(gt_dist,pre_dist)
            # corr_loss = corr_loss_method(img_feat_0,pc_feat_0.unsqueeze(-1),img_features,pc_features.unsqueeze(-1))
            
            if loss_reg.item() < 8:
                loss_all= loss_desc+loss_det+loss_epro+loss_reg+loss_corr
            else:
                loss_all= loss_desc+loss_det+loss_corr
                loss_epro, loss_reg = loss_desc, loss_det # for print

            loss_all.backward()
            with torch.no_grad():
                torch.distributed.all_reduce(loss_all)
                loss_all = loss_all / torch.tensor(args.world_size)
                
            optimizer.step()
            scheduler.step()
            
            if global_step%6==0 and args.rank == 0:
                # logger.info('%s-%d-%d, loss: %f, loss desc: %f, loss det: %f, loss epro: %f, loss reg: %f'%('train',epoch,global_step,loss_all.data.cpu().numpy(),loss_desc.data.cpu().numpy(),loss_det.data.cpu().numpy(), loss_epro.data.cpu().numpy(), loss_reg.data.cpu().numpy()))
                # print('%s-%d-%d, loss: %f, loss desc: %f, loss det: %f, loss epro: %f, loss reg: %f'%('train',epoch,global_step,loss_all.data.cpu().numpy(),loss_desc.data.cpu().numpy(),loss_det.data.cpu().numpy(), loss_epro.data.cpu().numpy(), loss_reg.data.cpu().numpy()))
                logger.info('%s-%d-%d, loss: %f, loss desc: %f, loss det: %f, loss epro: %f, loss reg: %f, loss corr: %f'%('train',epoch,global_step,loss_all.data.cpu().numpy(),loss_desc.data.cpu().numpy(),loss_det.data.cpu().numpy(), loss_epro.data.cpu().numpy(), loss_reg.data.cpu().numpy(),loss_corr.item()))
                print('%s-%d-%d, loss: %f, loss desc: %f, loss det: %f, loss epro: %f, loss reg: %f, loss corr: %f'%('train',epoch,global_step,loss_all.data.cpu().numpy(),loss_desc.data.cpu().numpy(),loss_det.data.cpu().numpy(), loss_epro.data.cpu().numpy(), loss_reg.data.cpu().numpy(),loss_corr.item()))
                
            if global_step%args.val_freq==0 and epoch>5 and args.rank == 0:
                # torch.save(model.state_dict(),os.path.join(logdir,'mode_last.t7'))
                torch.save(model.module.state_dict(),os.path.join(logdir,'mode_last.t7'))
            # if epoch>10 and args.rank == 0:
            #     # torch.save(model.state_dict(),os.path.join(logdir,'mode_last.t7'))
            #     torch.save(model.module.state_dict(),os.path.join(logdir,'mode_loss_best.t7'))

        if epoch%1==0 and args.rank == 0:
            current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
            logger.info('%s-%d-%d, current lr is %f'%('train',epoch,global_step,current_lr))
            print('%s-%d-%d, current lr is %f'%('train',epoch,global_step,current_lr))
        if args.rank == 0:
            # torch.save(model.state_dict(),os.path.join(logdir,'mode_epoch_%d.t7'%epoch))
            torch.save(model.module.state_dict(),os.path.join(logdir,'mode_epoch_%d.t7'%epoch))
    torch.distributed.destory_process_group()
        
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train_kitti_corr_360_multi_gpu.py
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train_kitti_corr_360_multi_gpu.py --load_ckpt='mode_epoch_16.t7'