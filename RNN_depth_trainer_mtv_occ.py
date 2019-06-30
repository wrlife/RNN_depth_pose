from data.data_loader_outdoor import *
from model import *
import time
from utils_lr import *


class RNN_depth_trainer:
    '''
    A wrapper class which create a dataloader, construct a network model and compute loss
    '''
    # ========================
    # Construct data loader
    # ========================
    def initDataloader(self,
                       dataset_dir,
                       batch_size=3,
                       img_height=128, #192,#
                       img_width=416, #256,#
                       num_views=10,
                       num_epochs=50,
                       is_training=True):

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_views = num_views
        self.num_epochs = num_epochs

        # Initialize data loader
        initloader = DataLoader(dataset_dir,
                                batch_size,
                                img_height,
                                img_width,
                                num_epochs,
                                self.num_views)

        dataLoader = initloader.inputs(is_training)

        return dataLoader

    def load_data(self, dataLoader):
        '''
        Load a single data sample
        '''
        with tf.device(None):
            data_dict = dataLoader.get_next()
        return data_dict

    # ========================
    # Construct model
    # ========================
    def construct_model(self, data_dict):

        # ------------------
        # Forward
        # ------------------
        image_seq = data_dict['image_seq']
        hidden_state = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                        None]  # Initialize hidden state
        hidden_state_pose = [None, None, None, None, None, None, None]
        est_poses = []

        for i in range(self.num_views):

            image = tf.slice(image_seq,
                             [0, 0, self.img_width * i, 0],
                             [-1, -1, int(self.img_width), -1])

            image.set_shape([self.batch_size, self.img_height, self.img_width, 3])

            # Depth prediction
            pred_depth, hidden_state = rnn_depth_net_encoderlstm(image, hidden_state, is_training=True)
            # Pose prediction
            pred_pose, hidden_state_pose = pose_net(tf.concat([image,pred_depth],axis=3), hidden_state_pose, is_training=True)
            est_poses.append(pred_pose)

            if i == 0:
                est_depths = pred_depth
            else:
                est_depths = tf.concat([est_depths, pred_depth], axis=2)


        #------------------
        # Backward
        #------------------
        image_seq = data_dict['image_seq']
        hidden_state = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None] # Initialize hidden state
        hidden_state_pose = [None,None,None,None,None,None,None]
        est_poses_bw = []

        for i in range(self.num_views-1,-1,-1):

            image =  tf.slice(image_seq,
                                [0, 0, self.img_width*i, 0],
                                [-1, -1, int(self.img_width), -1])
            depth = tf.slice(data_dict['depth_seq'],
                           [0, 0, self.img_width * i, 0],
                           [-1, -1, int(self.img_width), -1])

            image.set_shape([self.batch_size, self.img_height, self.img_width, 3])

            # Depth prediction
            pred_depth, hidden_state = rnn_depth_net_encoderlstm(image, hidden_state,is_training=True)
            # Pose prediction
            pred_pose, hidden_state_pose = pose_net(tf.concat([image,pred_depth],axis=3), hidden_state_pose,is_training=True)
            est_poses_bw.append(pred_pose)

            if i==self.num_views-1:
                est_depths_bw = pred_depth
                depth_seq_bw = depth
            else:
                est_depths_bw = tf.concat([est_depths_bw,pred_depth],axis = 2)
                depth_seq_bw = tf.concat([depth_seq_bw,depth],axis = 2)
        data_dict['depth_seq_bw'] = depth_seq_bw

        return [est_depths, est_poses, est_depths_bw, est_poses_bw] #

    # ========================
    # Compute loss
    # ========================
    def compute_loss(self, estimates,  data_dict, global_step): #

        est_depths = estimates[0]
        est_poses = estimates[1]
        est_depths_bw = estimates[2]
        est_poses_bw = estimates[3]

        all_losses = []  # keep different losses and visualize in tensorboard
        output_dict = {}

        def l1loss(label, pred, v_weight=None):
            # diff = tf.abs(label - pred)
            # diff = tf.where(tf.is_inf(diff), tf.zeros_like(diff), diff)
            # diff = tf.where(tf.is_nan(diff), tf.zeros_like(diff), diff)
            # # div = tf.count_nonzero(diff,dtype=tf.float32)
            # if v_weight is not None:
            #     diff = tf.multiply(diff, v_weight)

            # if v_weight is not None:
            #     return tf.reduce_sum(diff)/(tf.count_nonzero(v_weight,dtype=tf.float32)+0.000000001)
            # else:
            #     return tf.reduce_mean(diff)
            diff = tf.abs(label - pred)
            diff = tf.where(tf.is_inf(diff), tf.zeros_like(diff), diff)
            diff = tf.where(tf.is_nan(diff), tf.zeros_like(diff), diff)
            div = tf.count_nonzero(diff,dtype=tf.float32)
            if v_weight is not None:
                diff = tf.multiply(diff, v_weight)
            if v_weight is not None:
                return tf.reduce_sum(diff)/(tf.count_nonzero(v_weight,dtype=tf.float32)+0.000000001)
            else:
                return tf.reduce_sum(diff)/(div+0.000000001)

        def smooth_l1_loss(y_true, y_pred, v_weight=None):
            """Implements Smooth-L1 loss.
            y_true and y_pred are typically: [N, 4], but could be any shape.
            """
            diff = tf.abs(y_true - y_pred)
            diff = tf.where(tf.is_inf(diff), tf.zeros_like(diff), diff)
            diff = tf.where(tf.is_nan(diff), tf.zeros_like(diff), diff)
            if v_weight is not None:
                diff = tf.multiply(diff, v_weight)
            less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
            loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

            if v_weight is not None:
                return tf.reduce_sum(loss)/(tf.count_nonzero(v_weight,dtype=tf.float32)+0.000000001)
            else:
                return tf.reduce_mean(loss)

        def gradient(pred, delta):
            D_dx = pred[:, delta:, :, :] - pred[:, :-delta, :, :]
            D_dy = pred[:, :, delta:, :] - pred[:, :, :-delta, :]

            return D_dx, D_dy

        def compute_smooth_loss(img, disp, delta):
            disp_gradients_x, disp_gradients_y = gradient(disp, delta)

            image_gradients_x, image_gradients_y = gradient(img, delta)

            weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
            weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

            smoothness_x = disp_gradients_x * weights_x
            smoothness_y = disp_gradients_y * weights_y

            return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

        def SSIM( x, y):
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
            mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

            sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
            sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
            sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

            SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

            SSIM = SSIM_n / SSIM_d

            return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


        def image_similarity(x, y, v_weight=None, alpha=1.0):

            ssim = tf.image.ssim(x, y, max_val=1.0)
            diff = alpha * (1 - ssim) / 2  #+ (1-alpha) * tf.abs(x-y) #SSIM(x, y)

            if v_weight is not None:
                diff = tf.multiply(diff, v_weight)

            if v_weight is not None:
                return tf.reduce_sum(diff)/(tf.count_nonzero(v_weight,dtype=tf.float32)+0.000000001)
            else:
                return tf.reduce_mean(diff)


        # --------------------
        # Direct inverse depth loss
        # --------------------
        depth_loss_fw = tf.zeros([], dtype=tf.float32)
        gt_depth_fw = 1.0/data_dict['depth_seq']
        depth_loss_fw = l1loss(est_depths, gt_depth_fw)
        all_losses.append(depth_loss_fw*10)

        output_dict['depth'] = est_depths

        ####################
        #Backward depth
        ####################
        depth_loss_bw = tf.zeros([], dtype=tf.float32)
        gt_depth_bw = 1.0/data_dict['depth_seq_bw']
        depth_loss_bw = l1loss(est_depths_bw, gt_depth_bw)
        all_losses.append(depth_loss_bw*10)

        output_dict['depth_bw'] = est_depths_bw

        # --------------------
        # Gradient loss
        # --------------------
        gradient_loss_fw = tf.zeros([], dtype=tf.float32)

        for i in range(self.num_views):
            depth_slice = tf.slice(est_depths,
                                   [0, 0, self.img_width * i, 0],
                                   [-1, -1, int(self.img_width), -1])

            gt_depth_slice = tf.slice(gt_depth_fw,
                                     [0,0,self.img_width*i,0],
                                     [-1,-1,int(self.img_width),-1])

            image_slice = tf.slice(data_dict['image_seq'],
                                   [0, 0, self.img_width * i, 0],
                                   [-1, -1, int(self.img_width), -1])


            # gradient_loss_fw += compute_smooth_loss(image_slice, depth_slice, 1)
            # gradient_loss_fw += compute_smooth_loss(image_slice, depth_slice, 2)
            # gradient_loss_fw += compute_smooth_loss(image_slice, depth_slice, 4)

            Dx1, Dy1 = gradient(depth_slice,1); gtDx1, gtDy1 = gradient(gt_depth_slice,1)
            Dx2, Dy2 = gradient(depth_slice,2); gtDx2, gtDy2 = gradient(gt_depth_slice,2)
            Dx4, Dy4 = gradient(depth_slice,4); gtDx4, gtDy4 = gradient(gt_depth_slice,4)
            Dx8, Dy8 = gradient(depth_slice,8); gtDx8, gtDy8 = gradient(gt_depth_slice,8)

            gradient_loss_fw += (l1loss(Dx1, gtDx1)+
                           l1loss(Dy1, gtDy1))
            gradient_loss_fw += (l1loss(Dx2, gtDx2)+
                           l1loss(Dy2, gtDy2))
            gradient_loss_fw += (l1loss(Dx4, gtDx4)+
                           l1loss(Dy4, gtDy4))
            gradient_loss_fw += (l1loss(Dx8, gtDx8)+
                           l1loss(Dy8, gtDy8))

        all_losses.append(gradient_loss_fw*20)

        ####################
        #Backward gradient
        ####################
        gradient_loss_bw = tf.zeros([], dtype=tf.float32)

        for i in range(self.num_views):
            depth_slice = tf.slice(est_depths_bw,
                                  [0, 0, self.img_width * i, 0],
                                  [-1, -1, int(self.img_width), -1])
            gt_depth_slice = tf.slice(gt_depth_bw,
                                    [0,0,self.img_width*i,0],
                                    [-1,-1,int(self.img_width),-1])
            image_slice = tf.slice(data_dict['image_seq'],
                                  [0, 0, self.img_width * (self.num_views-i-1), 0],
                                  [-1, -1, int(self.img_width), -1])


            # gradient_loss_bw += compute_smooth_loss(image_slice, depth_slice, 1)
            # gradient_loss_bw += compute_smooth_loss(image_slice, depth_slice, 2)
            # gradient_loss_bw += compute_smooth_loss(image_slice, depth_slice, 4)

            Dx1, Dy1 = gradient(depth_slice,1); gtDx1, gtDy1 = gradient(gt_depth_slice,1)
            Dx2, Dy2 = gradient(depth_slice,2); gtDx2, gtDy2 = gradient(gt_depth_slice,2)
            Dx4, Dy4 = gradient(depth_slice,4); gtDx4, gtDy4 = gradient(gt_depth_slice,4)
            Dx8, Dy8 = gradient(depth_slice,8); gtDx8, gtDy8 = gradient(gt_depth_slice,8)

            gradient_loss_bw += (l1loss(Dx1, gtDx1)+
                           l1loss(Dy1, gtDy1))
            gradient_loss_bw += (l1loss(Dx2, gtDx2)+
                           l1loss(Dy2, gtDy2))
            gradient_loss_bw += (l1loss(Dx4, gtDx4)+
                           l1loss(Dy4, gtDy4))
            gradient_loss_bw += (l1loss(Dx8, gtDx8)+
                           l1loss(Dy8, gtDy8))

        all_losses.append(gradient_loss_bw*20)



        # --------------------
        # Pose loss, pose loss is
        # imposed by image reconstruction loss
        # --------------------

        pose_loss_fw = tf.zeros([], dtype=tf.float32)
        mask_loss_fw = tf.zeros([], dtype=tf.float32)

        mask_regular = tf.ones([self.batch_size, self.img_height, self.img_width, 1])

        # Backward pose
        pose_loss_bw = tf.zeros([], dtype=tf.float32)
        mask_loss_bw = tf.zeros([], dtype=tf.float32)


        consist_loss_fw = tf.zeros([], dtype=tf.float32)
        consist_loss_bw = tf.zeros([], dtype=tf.float32)

        #===================================
        # Multi-view image reprojection loss
        #===================================
        for i in range(self.num_views):

            if i==0:

                depth_slice_fw = tf.slice(est_depths,
                                       [0, 0, self.img_width * i, 0],
                                       [-1, -1, int(self.img_width), -1])
                depth_slice_bw = tf.slice(est_depths_bw,
                                       [0, 0, self.img_width * i, 0],
                                       [-1, -1, int(self.img_width), -1])
                image_slice_fw = tf.slice(data_dict['image_seq'],
                                       [0, 0, self.img_width * i, 0],
                                       [-1, -1, int(self.img_width), -1])
                image_slice_bw = tf.slice(data_dict['image_seq'],
                                       [0, 0, self.img_width * (self.num_views-1), 0],
                                       [-1, -1, int(self.img_width), -1])
                depth_slice_fw.set_shape([self.batch_size, self.img_height, self.img_width, 1])
                depth_slice_bw.set_shape([self.batch_size, self.img_height, self.img_width, 1])
                image_slice_fw.set_shape([self.batch_size, self.img_height, self.img_width, 3])
                image_slice_bw.set_shape([self.batch_size, self.img_height, self.img_width, 3])
                proj_img, wmask, flow_fw = projective_inverse_warp(
                    image_slice_fw,
                    tf.squeeze( 1.0/depth_slice_fw, axis=3),
                    est_poses[i],
                    data_dict['intrinsics'],
                    format='eular'
                )
                pose_loss_fw += image_similarity(image_slice_fw, proj_img, wmask)*0.1
                mask_loss_fw +=l1loss(mask_regular, wmask)*0.1

                proj_img, wmask, flow_bw = projective_inverse_warp(
                    image_slice_bw,
                    tf.squeeze( 1.0/depth_slice_bw, axis=3),
                    est_poses_bw[i],
                    data_dict['intrinsics'],
                    format='eular'
                )
                pose_loss_bw += image_similarity(image_slice_bw, proj_img, wmask)*0.1
                mask_loss_bw += l1loss(mask_regular, wmask)*0.1

                # visualize first depth map
                if i==0:
                    depth_slice_fw1 = depth_slice_fw

                continue

            # Get current image and depth
            depth_slice_fw = tf.slice(est_depths,
                                   [0, 0, self.img_width * i, 0],
                                   [-1, -1, int(self.img_width), -1])


            image_slice_fw = tf.slice(data_dict['image_seq'],
                                   [0, 0, self.img_width * i, 0],
                                   [-1, -1, int(self.img_width), -1])


            depth_slice_fw.set_shape([self.batch_size, self.img_height, self.img_width, 1])
            image_slice_fw.set_shape([self.batch_size, self.img_height, self.img_width, 3])


            accum_pose_fw = tf.eye(4,4,[self.batch_size])
            accum_pose_bw = tf.eye(4,4,[self.batch_size])

            if i==4:
                depth_slice_fw5 = depth_slice_fw

            degrade = 0

            # Project current view into every previous view using accumulate
            # transformation
            for j in range(i-1, -1, -1):

                previous_scene_fw = tf.slice(data_dict['image_seq'],
                                       [0, 0, self.img_width * j, 0],
                                       [-1, -1, int(self.img_width), -1])
                previous_scene_fw.set_shape([self.batch_size, self.img_height, self.img_width, 3])

                accum_pose_fw = tf.matmul(pose_vec2mat(est_poses[j+1],'eular'), accum_pose_fw)

                # Differentiable geometric module (DGM)
                # Using depth and pose to compute warped image, and warping flow
                proj_img_fw, wmask_fw, flow_fw = projective_inverse_warp(
                    previous_scene_fw,
                    tf.squeeze( 1.0/depth_slice_fw, axis=3),
                    accum_pose_fw,
                    data_dict['intrinsics'],
                    format='matrix'
                )

                # Perform the same operation for backward prediction
                depth_slice_bw = tf.slice(est_depths_bw,
                                       [0, 0, self.img_width * (self.num_views-j-1), 0],
                                       [-1, -1, int(self.img_width), -1])
                depth_slice_bw.set_shape([self.batch_size, self.img_height, self.img_width, 1])

                accum_pose_bw = tf.matmul(accum_pose_bw, pose_vec2mat(est_poses_bw[self.num_views-j-1],'eular'))

                proj_img_bw, wmask_bw, flow_bw = projective_inverse_warp(
                    image_slice_fw,
                    tf.squeeze( 1.0/depth_slice_bw, axis=3),
                    accum_pose_bw,
                    data_dict['intrinsics'],
                    format='matrix'
                )

                #------------------------------
                # Compute Forward backward flow
                #-----------------------------
                Fba_i, _, _ = projective_inverse_warp(
                                  -flow_bw,
                                  tf.squeeze(1.0/depth_slice_fw, axis=3),
                                  accum_pose_fw,
                                  data_dict['intrinsics'],
                                  format='matrix'
                                )

                Fab_i, _, _ = projective_inverse_warp(
                                  -flow_fw,
                                  tf.squeeze(1.0 / depth_slice_bw, axis=3),
                                  accum_pose_bw,
                                  data_dict['intrinsics'],
                                  format='matrix'
                                )

                # Find inconsistency
                fw_flowdiff = tf.norm(flow_fw - Fba_i, axis=3)
                bw_flowdiff = tf.norm(flow_bw - Fab_i, axis=3)

                # A threshold to filter out moving object or occlusion boundary
                fw_threshold_flow = tf.where(tf.less_equal(0.05*tf.norm(flow_fw, axis=3),3.0), tf.ones_like(fw_flowdiff)*3.0, 0.05*tf.norm(flow_fw, axis=3))
                bw_threshold_flow = tf.where(tf.less_equal(0.05*tf.norm(flow_bw, axis=3),3.0),  tf.ones_like(fw_flowdiff)*3.0, 0.05*tf.norm(flow_bw, axis=3))
                fw_occ_mask = tf.expand_dims(tf.where(tf.less_equal(fw_flowdiff,fw_threshold_flow), tf.ones_like(fw_flowdiff), tf.zeros_like(fw_flowdiff)),axis=-1)
                bw_occ_mask = tf.expand_dims(tf.where(tf.less_equal(bw_flowdiff,bw_threshold_flow), tf.ones_like(fw_flowdiff), tf.zeros_like(fw_flowdiff)),axis=-1)

                # A mask for valid pixels.
                wmask_fw = wmask_fw*fw_occ_mask
                wmask_bw = wmask_bw*bw_occ_mask

                # Image reprojection loss
                pose_loss_fw += image_similarity(image_slice_fw, proj_img_fw, wmask_fw)/2**degrade/10.0
                pose_loss_bw += image_similarity(previous_scene_fw, proj_img_bw, wmask_bw)/2**degrade/10.0

                #==================================
                # Foward back flow consistency loss
                #==================================
                if j == i-1:
                    consist_fw_err = l1loss(flow_fw, Fba_i, wmask_fw[:,:,:,0:2])
                    consist_loss_fw += consist_fw_err*0.01/2**degrade
                    consist_bw_err = l1loss(flow_bw, Fab_i, wmask_bw[:,:,:,0:2])
                    consist_loss_bw += consist_bw_err*0.01/2**degrade

                    # A regularization loss on mask to prevent trivial solution
                    mask_loss_fw +=l1loss(mask_regular, wmask_fw)*0.01/2**degrade
                    mask_loss_bw +=l1loss(mask_regular, wmask_bw)*0.01/2**degrade

                    flow_fw_img = flow_fw

                if i==1:
                    flow_bw_img = flow_bw

                degrade+=1

        all_losses.append(pose_loss_fw)
        all_losses.append(pose_loss_bw)
        data_dict['est_pose'] = est_poses
        data_dict['est_pose_bw'] = est_poses_bw

        tf.summary.image('proj_img_fw', tf.concat([previous_scene_fw, proj_img_fw, image_slice_fw], axis=1), max_outputs=1)
        tf.summary.image('proj_img_bw', tf.concat([image_slice_fw, proj_img_bw, previous_scene_fw], axis=1), max_outputs=1)
        tf.summary.image('fw_occ_mask',fw_occ_mask, max_outputs=1)
        tf.summary.image('bw_occ_mask', bw_occ_mask, max_outputs=1)


        all_losses.append(mask_loss_fw)
        all_losses.append(mask_loss_bw)
        all_losses.append(consist_loss_fw)
        all_losses.append(consist_loss_bw)


        # Sum up all different losses
        # TODO find balance
        total_loss = tf.reduce_mean(all_losses)

        output_dict['previous_scene_fw'] = previous_scene_fw
        output_dict['proj_img_fw'] = proj_img_fw
        output_dict['image_slice_fw'] = image_slice_fw
        output_dict['depth_slice_fw'] = depth_slice_fw

        output_dict['previous_scene_bw'] = image_slice_fw
        output_dict['proj_img_bw'] = proj_img_bw
        output_dict['image_slice_bw'] = previous_scene_fw
        output_dict['depth_slice_bw'] = depth_slice_bw

        output_dict['flow_bw'] = flow_bw_img
        output_dict['flow_fw'] = flow_fw_img

        data_dict['flow_bw'] = flow_bw
        data_dict['flow_fw'] = flow_fw

        output_dict['depth_slice_fw1'] = depth_slice_fw1
        output_dict['depth_slice_fw5'] = depth_slice_fw5

        return total_loss, all_losses, output_dict

    # ========================
    # Tensorflow summary
    # ========================
    def sub_depth(self, est_depths):
        depth_slice1 = tf.slice(est_depths,
                               [0, 0, self.img_width * 0, 0],
                               [-1, -1, int(self.img_width), -1])
        depth_slice2 = tf.slice(est_depths,
                               [0, 0, self.img_width * 5, 0],
                               [-1, -1, int(self.img_width), -1])
        depth_slice3 = tf.slice(est_depths,
                               [0, 0, self.img_width * 9, 0],
                               [-1, -1, int(self.img_width), -1])
        depth = 1.0/tf.concat([depth_slice1,depth_slice2,depth_slice3],axis=2)
        return depth

    def tb_summary(self, output_dict, loss, all_losses):


        # flow_bw = tf.expand_dims(tf.py_func(flow_to_image, [output_dict['flow_bw'][0,:,:,:]], tf.uint8),axis=0)
        # tf.summary.image('flow_bw', flow_bw)
        # flow_fw = tf.expand_dims(tf.py_func(flow_to_image, [output_dict['flow_fw'][0,:,:,:]], tf.uint8),axis=0)
        # tf.summary.image('flow_fw', flow_fw)


        color_depth_fw = tf.expand_dims(tf.py_func(depth_plasma, [output_dict['depth_slice_fw'][0,:self.img_height-10,:,0]], tf.float32),axis=0)
        color_depth_bw = tf.expand_dims(tf.py_func(depth_plasma, [output_dict['depth_slice_bw'][0,:self.img_height-10,:,0]], tf.float32),axis=0)

        tf.summary.image('color_depth_fw10', color_depth_fw)
        tf.summary.image('color_depth_bw10', color_depth_bw)


        color_depth_fw1 = tf.expand_dims(tf.py_func(depth_plasma, [output_dict['depth_slice_fw1'][0,:self.img_height-10,:,0]], tf.float32),axis=0)
        color_depth_fw5 = tf.expand_dims(tf.py_func(depth_plasma, [output_dict['depth_slice_fw5'][0,:self.img_height-10,:,0]], tf.float32),axis=0)

        tf.summary.image('color_depth_fw1', color_depth_fw1)
        tf.summary.image('color_depth_fw5', color_depth_fw5)

        depth = self.sub_depth(output_dict['depth'])
        depth_bw = self.sub_depth(output_dict['depth_bw'])



        depth = self.sub_depth(output_dict['depth'])
        depth_bw = self.sub_depth(output_dict['depth_bw'])

        tf.summary.scalar('losses/total_loss', loss)
        tf.summary.image('est_depth_fw', depth)

        # Distinguish different losses
        tf.summary.scalar('losses/depth_fw', all_losses[0])
        tf.summary.scalar('losses/depth_bw', all_losses[1])

        tf.summary.scalar('losses/depth_gradient_fw', all_losses[2])
        tf.summary.scalar('losses/depth_gradient_bw', all_losses[3])

        tf.summary.scalar('losses/pose_fw', all_losses[4])
        tf.summary.scalar('losses/pose_bw', all_losses[5])

        tf.summary.scalar('losses/mask_fw', all_losses[6])
        tf.summary.scalar('losses/mask_bw', all_losses[7])

        tf.summary.scalar('losses/consist_fw', all_losses[8])
        tf.summary.scalar('losses/consist_bw', all_losses[9])


    # ========================
    # Run session
    # ========================
    def save(self, sess, checkpoint_dir, step, saver):
        '''
        Save checkpoints
        '''
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            saver.save(sess,
                       os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            saver.save(sess,
                       os.path.join(checkpoint_dir, model_name),
                       global_step=step)


    def train(self, train_op, avg_loss, eval_step, args, data_dict):

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # Saver has all the trainable parameters
        saver = tf.train.Saver()
        #saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rnn_depth_net"))
        #saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pose_net"))

        total_time = 0.0
        # Session start
        with tf.Session(config=config) as sess:

            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(args.checkpoint_dir + '/logs/train',
                                                 sess.graph)
            eval_writer = tf.summary.FileWriter(args.checkpoint_dir + '/logs/eval')
            merged = tf.summary.merge_all()

            # Load parameters
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            with tf.name_scope("parameter_count"):
                parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                                 for v in tf.trainable_variables()])
            print("parameter_count =", sess.run(parameter_count))

            # Restore model
            if args.continue_train == True:
                saver.restore(sess, tf.train.latest_checkpoint(args.restore_path))

            try:
                step = 0
                while True:
                    start_time = time.time()
                    fetches = {
                        "loss": avg_loss,
                        "summary": merged,
                        "data_dict": data_dict
                    }

                    if step % args.eval_freq == 0 and args.eval_set_dir is not None:
                        do_eval = True
                    else:
                        do_eval = False
                        fetches["train_op"] = train_op

                    results = sess.run(fetches, feed_dict={eval_step: do_eval})

                    duration = time.time() - start_time

                    total_time += duration

                    if step % args.eval_freq == 0 and args.eval_set_dir is not None:
                        print('Step %d: eval loss = %.5f (%.3f sec)' % (step,
                                                                        results["loss"],
                                                                        duration))
                        print(results['data_dict']['est_pose'][5])
                        print(results['data_dict']['est_pose_bw'][5])
                        eval_writer.add_summary(results["summary"], step)

                    elif step % args.summary_freq == 0:
                        print('Step %d: loss = %.5f (%.3f sec)' % (step,
                                                                   results["loss"],
                                                                   duration))
                        train_writer.add_summary(results["summary"], step)


                    # Save latest model
                    if step % args.save_latest_freq == 0:
                        self.save(sess, args.checkpoint_dir, step, saver)

                    step += 1

            except tf.errors.OutOfRangeError:
                print('Total time: %f' % total_time)
                print('Done training for %d epochs, %d steps.' % (self.num_epochs,
                                                                  step))


