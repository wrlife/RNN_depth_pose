import math
import itertools
import numpy as np
import pykitti
import sys
from collections import Counter, namedtuple
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from PIL import Image
# from scipy.ndimage import maximum_filter1d, minimum_filter1d
#
# from .view import View, View_kitti
# from .view_tools import *
# from .helpers import measure_sharpness
# import cv2
# from scipy import interpolate
#
# from .util import *

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def long_image_kitti(views, max_views_num):
    height = views[0].image.shape[0]
    width = views[0].image.shape[1]
    total_width = width * max_views_num
    new_im = np.zeros((height, total_width, 3), dtype=np.uint8)
    new_depth = np.zeros((height, total_width), dtype=np.float32)
    new_motion = np.zeros((4, 4 * max_views_num), dtype=np.float32)
    x_offset = 0
    RT_offset = 0

    for view in views:
        new_im[:, x_offset:x_offset + width, :] = view.image
        new_depth[:, x_offset:x_offset + width] = view.depth
        x_offset += width

        new_motion[:, RT_offset:RT_offset + 4] = np.reshape(view.P, (4, 4))
        RT_offset += 4

    return new_im, new_depth, new_motion

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(P_velo2im, velo, im_shape, interp=False, vel_depth=False):

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    #import pdb;pdb.set_trace()
    velo = velo[velo[:, 2] >= 0, :]
    velo[:, 3] = 1.0

    #import pdb;pdb.set_trace()
    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T

    velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    #velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
    #velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:,0] < im_shape[1]) & (velo_pts_im[:,1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]


    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds==dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth<0] = 0

    if interp:
        # interpolate the depth map to fill in holes
        depth_interp = lin_interp(im_shape, velo_pts_im)
        #plt.imsave("test.png", depth_interp, cmap='plasma')
        return depth, depth_interp
    else:
        return depth



def get_image_grid(width,height,fx,fy,cx,cy):
    return np.meshgrid(
      (np.arange(width)  - cx) / fx,
      (np.arange(height) - cy) / fy)


def generate_surface(z, width, height,fx,fy, cx, cy):
      S = np.dstack((get_image_grid(width, height,fx,fy, cx, cy) + [np.ones_like(z)])) * z[:,:,np.newaxis]
      pad = np.ones_like(z)
      return np.dstack([S]+[pad]).reshape([-1,4])






def read_image_depth_from_idx(dataset,idx,resizedwidth,resizedheight):
    image = cv2.resize(np.array(dataset.get_cam3(idx)),(resizedwidth,resizedheight))
    #velo = dataset.get_velo(idx)
    pose =  np.dot(dataset.calib.T_cam3_imu, dataset.oxts[idx].T_w_imu)
    #import pdb;pdb.set_trace()
    #depth = generate_depth_map(np.dot(dataset.calib.P_rect_20,dataset.calib.T_cam2_velo), velo, image.shape[:2])
    return image,pose#,depth

def create_samples_from_sequence_kitti(tfrecordfile, kitti_path, depth_path, seq_name, max_views_num=10):
    """Read a KITTI sequence and write samples to the tfrecordfile

    tfrecordfile: tensorflow data format

    kitti_path: str
        base path to the kitti data

    seq_name: str
        the name of the sequence e.g. "20xx_xx_xx_sync"

    Returns the number of generated groups
    """

    writer = tf.python_io.TFRecordWriter(tfrecordfile)
    resizedheight = 128
    resizedwidth = 416
    generated_groups = 0

    date = seq_name[:10]
    drive = seq_name[-9:-5]

    dataset = pykitti.raw(kitti_path, date, drive)

    #A tuple to store information for each view
    View_kitti = namedtuple('View', {'P', 'K', 'image', 'depth'})

    # read intrinsics
    intrinsics_ori = dataset.calib.K_cam3
    dataset.calib.P_rect_30[0,3] *= 0#(resizedwidth / 1600)
    dataset.calib.P_rect_30[1,3] *= 0#(resizedheight / 375)
    dataset.calib.P_rect_30[2,3] *= 0

    if len(dataset.velo_files)<=0:
    	return 0

    image = np.array(dataset.get_cam3(0))

    ori_height, ori_width = image.shape[:2]


    intrinsics = intrinsics_ori.copy()
    intrinsics[0, 0] = intrinsics_ori[0, 0] * resizedwidth / ori_width
    intrinsics[0, 2] = intrinsics_ori[0, 2] * resizedwidth / ori_width
    intrinsics[1, 1] = intrinsics_ori[1, 1] * resizedheight / ori_height
    intrinsics[1, 2] = intrinsics_ori[1, 2] * resizedheight / ori_height

    import pdb;pdb.set_trace()

    homo_intrinsic = np.concatenate([intrinsics,np.zeros([3,1])],axis=1)
    homo_intrinsic = np.concatenate([homo_intrinsic,np.zeros([1,4])],axis=0)
    homo_intrinsic[3,3]=1.0
    mean_baseline = []

    for idx in range(len(dataset.velo_files)):


        file = dataset.cam3_files[idx].split('/')[-1]
        depth_file = os.path.join(depth_path,seq_name,'proj_depth','groundtruth','image_03',file)
        if not os.path.isfile(depth_file):
            continue

        image, pose = read_image_depth_from_idx(dataset,idx, resizedwidth, resizedheight)
        #import pdb;pdb.set_trace()
        depth = depth_read(depth_file)
        S = generate_surface(depth,ori_width,ori_height,intrinsics_ori[0, 0],intrinsics_ori[1, 1],intrinsics_ori[0, 2],intrinsics_ori[1, 2])
        depth = generate_depth_map(homo_intrinsic, S, image.shape[:2])

        # import pdb;pdb.set_trace()
        # plt.imsave("image2.png", image)
        # plt.imsave("depth2.png", depth, cmap='plasma')

        view1 = View_kitti(P=pose, K=intrinsics, image=image, depth=depth)
        views = [view1]

        T_pre = pose[0:3,3]
        R_pre = pose[0:3,0:3]

        #If there is no more than 10 images afterwards, we stop
        if(idx+9>=len(dataset.velo_files)):
            break

        for idx2 in range(idx+1,len(dataset.velo_files)):

            file = dataset.cam3_files[idx2].split('/')[-1]
            depth_file = os.path.join(depth_path,seq_name,'proj_depth','groundtruth','image_03',file)
            if not os.path.isfile(depth_file):
                continue

            image, pose = read_image_depth_from_idx(dataset,idx2, resizedwidth, resizedheight)
            depth = depth_read(depth_file)
            S = generate_surface(depth,ori_width,ori_height,intrinsics_ori[0, 0],intrinsics_ori[1, 1],intrinsics_ori[0, 2],intrinsics_ori[1, 2])
            depth = generate_depth_map(homo_intrinsic, S, image.shape[:2])

            #Check whether the scene is static
            T_curr = pose[0:3,3]
            R_curr = pose[0:3,0:3]
            baseline = np.linalg.norm((-R_pre.transpose().dot(T_pre)) - (-R_curr.transpose().dot(T_curr)))
            #import pdb;pdb.set_trace()
            if baseline < 0.3:
                continue

            mean_baseline.append(baseline)

            T_pre = T_curr
            R_pre = R_curr

            view2 = View_kitti(P=pose, K=intrinsics, image=image, depth=depth)
            views.append(view2)

            if len(views) == max_views_num:
                break

        if len(views)==max_views_num:

            #import pdb;pdb.set_trace()
            concat_view,concat_depth,concat_motion = long_image_kitti(views,max_views_num)
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_seq': _bytes_feature(concat_view.tostring()),
                'depth_seq': _bytes_feature(concat_depth.tostring()),
                'motion_seq': _bytes_feature(concat_motion.tostring()),
                'intrinsics': _bytes_feature(intrinsics.tostring()),
                }))

            writer.write(example.SerializeToString())
            generated_groups += 1

    print(np.mean(mean_baseline))
    writer.close()
    return generated_groups

# def main():
#     create_samples_from_sequence_kitti('test', '/playpen1/datasets/KITTI/KITTI_raw/', '2011_09_26_drive_0018_sync')

# if __name__ == "__main__":
#     sys.exit(main())

