import numpy as np
import os

# root_dir='../datas/kitti/poses'
# for i in range(11):
#     if not os.path.exists(os.path.join(root_dir,'%02d'%i)):
#         os.mkdir(os.path.join(root_dir,'%02d'%i))

#     data=np.loadtxt(os.path.join(root_dir,'%02d.txt'%i))
#     print(np.shape(data))
#     for k in range(np.shape(data)[0]):
#         T=data[k]
#         T=np.reshape(T,(3,4))
#         T=np.concatenate((T,np.array([[0,0,0,1]],dtype=np.float32)),axis=0)
#         # print(T)
#         # print(np.shape(T))
#         np.save(os.path.join(root_dir,'%02d'%i,'%06d.npy'%k),T)
#         print('sequences:%02d    %06d.npy'%(i, k))

a=np.load('../datas/kitti/poses/00/001001.npy')
print(a)