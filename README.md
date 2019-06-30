# RNN_depth_pose
This is a Tensorflow implementation of our paper:

Recurrent Neural Network for (Un-)supervised Learning of Monocular VideoVisual Odometry and Depth

Rui Wang, Stephen M. Pizer, Jan-Michael Frahm

arxiv preprint: (https://arxiv.org/abs/1904.07087)



## Prerequisites
This codebase was developed and tested with Python3.6 Tensorflow 1.12.0, CUDA 10.1 and Ubuntu 16.04.

## Preparing training data
Download [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) raw data and [depth](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php) data. Then process using the provided script in data/KITTI folder.

## Training
Once the data are formatted properly, you should be able to the model by running the following command

```
python main.py --dataset_dir=/path/to/tfrecords --checkpoint_dir=/path/to/output_checkpoints/
```

You can visualize training result using tensorboard

```
tensorboard --logdir=/path/to/output_checkpoints/ --port=8888
```

## TODO
The code will continue to be cleaned up and more comments will be added.

Unsupervised training version.

Demo with pretrained model will be added.

## Reference

```
@inproceedings{wang2019recurrent,
  title={Recurrent Neural Network for (Un-) supervised Learning of Monocular Video Visual Odometry and Depth},
  author={Wang, Rui and Pizer, Stephen M and Frahm, Jan-Michael},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5555--5564},
  year={2019}
}
```
