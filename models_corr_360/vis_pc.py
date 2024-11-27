import mayavi.mlab
import numpy as np
import os
 
 
def viz_mayavi(points, vals="distance"):  # 可视化只用到了3维数据(x,y,z)！
    x=points[:, 0]
    y=points[:, 1]
    z=points[:, 2]
    r=points[:, 3]  # reflectance value of point
    d=np.sqrt(x**2+y**2)
 
    if vals == "height":
        col = z
    else:
        col = d
    # 创建可视化模板的尺寸
    fig=mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    mayavi.mlab.points3d(x, y, z,
                         col,
                         mode="point",
                         colormap='spectral',
                         figure=fig,
                         )
 
    mayavi.mlab.show()

 
if __name__ == "__main__":
    bin_file_path = '/home/jlurobot/wzy/Calibration/Lidar2Camera/VP2P-Match-main/data/kitti/sequences/01/velodyne/000647.bin'
    bin_files = os.listdir(bin_file_path)
    for bin_file in bin_files:
        if bin_file.endswith(".bin"):
            mypointcloud = np.fromfile(bin_file_path + '/' + bin_file, dtype=np.float32, count=-1).reshape([-1, 4])
            viz_mayavi(mypointcloud,vals="height")