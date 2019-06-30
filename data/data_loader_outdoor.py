from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import os, glob
import cv2 as cv


def local_normalize_image(img):

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    float_gray = gray.astype(np.float32) / 255.0

    blur = cv.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur

    blur = cv.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv.pow(blur, 0.5)+0.0000001

    gray = num / den

    cv.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

    gray = np.concatenate((gray[:,:,np.newaxis],gray[:,:,np.newaxis],gray[:,:,np.newaxis]),axis=2)

    return gray
    

class DataLoader(object):
    """ Data loading class for training heatmap-attention-padding network

    Args:
        dataset_dir: Folder contain .tfrecords files
        batch_size: training batch size
        image_height, image_width: input image height and width
        opt: flags from input parser
    
    Returns:
        new_mask: A gauss smoothed tensor

    """

    def __init__(self,
                 dataset_dir,
                 batch_size,
                 image_height,
                 image_width,
                 num_epochs,
                 num_views):
        self.dataset_dir=dataset_dir
        self.batch_size=batch_size
        self.image_height=image_height
        self.image_width=image_width
        self.num_epochs = num_epochs
        self.num_views = num_views

    #==================================
    # Load training data from tf records
    #==================================
    def inputs(self, is_training=True):
        """Reads input data num_epochs times.
        Args:
            batch_size: Number of examples per returned batch.
            num_epochs: Number of times to read the input data, or 0/None to
            train forever.
        Returns:
            data_dict: A dictional contain input image and groundtruth label
        """
        def decode(serialized_example):
            """Parses an image and label from the given `serialized_example`."""
            features = tf.parse_single_example(
                serialized_example,
                # Defaults are not specified since both keys are required.
                features={
                    
                    'image_seq': tf.FixedLenFeature([], tf.string),
                    'depth_seq': tf.FixedLenFeature([], tf.string),
                    'intrinsics': tf.FixedLenFeature([], tf.string),
                })

            # Convert from a scalar string tensor (whose single string has
            # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
            # [mnist.IMAGE_PIXELS].

            image_seq = tf.decode_raw(features['image_seq'], tf.uint8)
            depth_seq = tf.decode_raw(features['depth_seq'], tf.float32)  #landmark visibility
            intrinsics = tf.decode_raw(features['intrinsics'], tf.float64)   #landmark coordinates

            image_seq_norm = tf.reshape(image_seq, 
                    [self.image_height, 
                    self.image_width*self.num_views, 3]) 
            image_seq_norm = tf.py_func(local_normalize_image, [image_seq_norm], tf.float32)
            image_seq_norm.set_shape([self.image_height,self.image_width*self.num_views,3])
            
            image_seq = tf.image.convert_image_dtype(
                tf.reshape(image_seq, 
                    [self.image_height, 
                    self.image_width*self.num_views, 3]),
                tf.float32)
            
            depth_seq = tf.cast(
                tf.reshape(depth_seq, 
                    [self.image_height, 
                    self.image_width*self.num_views, 1]),
                tf.float32)   
            
            
            intrinsics = tf.reshape(intrinsics,[3,3])

            data_dict = {}
            data_dict['image_seq'] = image_seq
            data_dict['image_seq_norm'] = image_seq_norm
            data_dict['depth_seq'] = depth_seq
            data_dict['intrinsics'] = tf.cast(intrinsics,tf.float32)
            

            # For testing, turn with_aug to false
            if is_training:
                data_dict = self.data_augmentation2(data_dict,self.image_height,self.image_width)

            return data_dict

        #If epochs number is none, then infinite repeat
        if not self.num_epochs:
            self.num_epochs = None
        
        #Get the list of training data
        filenames = glob.glob(os.path.join(self.dataset_dir,'*.tfrecords'))
        from random import shuffle
        shuffle(filenames)
        #import pdb;pdb.set_trace()
        
        with tf.name_scope('input'):
            dataset = tf.data.TFRecordDataset(filenames)
            # The map transformation takes a function and applies it to every element
            # of the dataset.
            if is_training:
                dataset = dataset.shuffle(100*self.batch_size) # when testing we do not shuffle data
            dataset = dataset.repeat(self.num_epochs)        # Number of epochs to train
            dataset = dataset.map(decode,num_parallel_calls=16)   
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(1)
            iterator = dataset.make_one_shot_iterator()

        return iterator#.get_next()


    def data_augmentation2(self, data_dict, out_h, out_w, is_training = True):

        def flip_intrinsics(intrinsics, width):
            fx = intrinsics[0,0]
            fy = intrinsics[1,1]
            cx = width-intrinsics[0,2]
            cy = intrinsics[1,2]
            
            zeros = tf.zeros_like(fx)
            r1 = tf.stack([fx, zeros, cx])
            r2 = tf.stack([zeros, fy, cy])
            r3 = tf.constant([0.,0.,1.])
            intrinsics = tf.stack([r1, r2, r3], axis=0)  

            return intrinsics

        def flip_left_right(image_seq, num_views):
            """Perform random distortions on an image.
            Args:
            image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
            thread_id: Preprocessing thread id used to select the ordering of color
              distortions. There should be a multiple of 2 preprocessing threads.
            Returns:
            distorted_image: A float32 Tensor of shape [height, width, 3] with values in
              [0, 1].
            """

            in_h, in_w, _ = image_seq.get_shape().as_list()
            in_w = in_w/num_views
            # Randomly flip horizontally.
                
            for i in range(num_views):
                # Scale up images
                image = tf.slice(image_seq,
                                 [0, int(in_w) * i, 0],
                                 [-1, int(in_w), -1])

                image = tf.image.flip_left_right(image)
    
    
                if i == 0:
                    flip_image = image
                else:
                    flip_image = tf.concat([flip_image, image],axis=1)
    
            
            return flip_image


        
        # Random scaling
        def random_scaling(data_dict, num_views):
        
            in_h, in_w, _ = data_dict['image_seq'].get_shape().as_list()
            
            in_w = in_w/num_views                
            
            scaling = tf.random_uniform([1], 1, 1.25)
            x_scaling = scaling[0]
            y_scaling = scaling[0]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            
            scaled_images = []
            scaled_depths = []
            scaled_images_norm = []
            
            for i in range(num_views):
            
                # Scale up images
                image = tf.slice(data_dict['image_seq'],
                                 [0, int(in_w) * i, 0],
                                 [-1, int(in_w), -1])
                image = tf.image.resize_images(image, [out_h, out_w])
                scaled_images.append(image)
                
                
                # Scale up normalized images
                image_norm = tf.slice(data_dict['image_seq_norm'],
                                 [0, int(in_w) * i, 0],
                                 [-1, int(in_w), -1])
                image_norm = tf.image.resize_images(image_norm, [out_h, out_w])
                scaled_images_norm.append(image_norm)
                
                
                # Scale up depth
                depth = tf.slice(data_dict['depth_seq'],
                                 [0, int(in_w) * i, 0],
                                 [-1, int(in_w), -1])
                depth = tf.image.resize_images(depth, [out_h, out_w],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                scaled_depths.append(depth)
            
            return scaled_images, scaled_depths, scaled_images_norm



        # Random cropping
        def random_cropping(data_dict, scaled_images, scaled_depths, scaled_images_norm, num_views, out_h, out_w):

            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            in_h, in_w, _ = tf.unstack(tf.shape(scaled_images[0]))
            
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]

            _in_h = tf.to_float(in_h)
            _in_w = tf.to_float(in_w)
            _out_h = tf.to_float(out_h)
            _out_w = tf.to_float(out_w)

            
            
            fx = data_dict['intrinsics'][0,0]*_in_w/_out_w
            fy = data_dict['intrinsics'][1,1]*_in_h/_out_h
            cx = data_dict['intrinsics'][0,2]*_in_w/_out_w-tf.cast(offset_x, tf.float32)
            cy = data_dict['intrinsics'][1,2]*_in_h/_out_h-tf.cast(offset_y, tf.float32)
            
            zeros = tf.zeros_like(fx)
            r1 = tf.stack([fx, zeros, cx])
            r2 = tf.stack([zeros, fy, cy])
            r3 = tf.constant([0.,0.,1.])
            data_dict['intrinsics'] = tf.stack([r1, r2, r3], axis=0)            

            for i in range(num_views):
            
                if i == 0:
                    cropped_images =  tf.image.crop_to_bounding_box(
                                        scaled_images[i], offset_y, offset_x, out_h, out_w)
                    cropped_depths =  tf.image.crop_to_bounding_box(
                                        scaled_depths[i], offset_y, offset_x, out_h, out_w)
                    cropped_images_norm =  tf.image.crop_to_bounding_box(
                                        scaled_images_norm[i], offset_y, offset_x, out_h, out_w)
                else:
                    cropped_images = tf.concat([cropped_images, tf.image.crop_to_bounding_box(
                                        scaled_images[i], offset_y, offset_x, out_h, out_w)], axis=1)
                    cropped_depths = tf.concat([cropped_depths, tf.image.crop_to_bounding_box(
                                        scaled_depths[i], offset_y, offset_x, out_h, out_w)], axis=1)
                    cropped_images_norm = tf.concat([cropped_images_norm, tf.image.crop_to_bounding_box(
                                        scaled_images_norm[i], offset_y, offset_x, out_h, out_w)], axis=1)                                       
            data_dict['image_seq'] = cropped_images    
            data_dict['depth_seq'] = cropped_depths 
            data_dict['image_seq_norm'] = cropped_images_norm
            
            return data_dict


        #data_dict=random_rotate(data_dict)
        # do_augment  = tf.random_uniform([], 0, 1)
        # data_dict['image_seq'] = tf.cond(do_augment > 0.5, lambda: flip_left_right(data_dict['image_seq'],self.num_views), lambda: data_dict['image_seq'])
        # data_dict['depth_seq'] = tf.cond(do_augment > 0.5, lambda: flip_left_right(data_dict['depth_seq'],self.num_views), lambda: data_dict['depth_seq'])
        # data_dict['intrinsics'] = tf.cond(do_augment > 0.5, lambda: flip_intrinsics(data_dict['intrinsics'],out_w), lambda: data_dict['intrinsics'])

        scaled_images, scaled_depths, scaled_images_norm = random_scaling(data_dict, self.num_views)
        data_dict = random_cropping(data_dict, scaled_images, scaled_depths, scaled_images_norm, self.num_views, out_h, out_w)

        return data_dict


