
import sys
import argparse
from RNN_depth_trainer_mtv_occ import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):

        PS_OPS = [
            'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
            'MutableHashTableOfTensors', 'MutableDenseHashTable'
        ]
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign


# Source:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main():

    # Add flags
    parser = argparse.ArgumentParser(description="Train RNN depth")
    parser.add_argument("--dataset_dir", type=str, default="/playpen1/rui/RNN_depth/data/KITTI/kitti_tfrecords_depth", help="The path to the data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="/playpen1/rui/RNN_depth/models/", help="The path to the checkpoint directory")
    parser.add_argument("--continue_train", type=bool, default=False, help="Continue train")
    parser.add_argument("--restore_path", type=str, default="", help="The path to load checkpoint")
    parser.add_argument("--eval_set_dir", type=str, default=None, help="The path to the evaluation directory")
    parser.add_argument("--num_epochs", type=int, default=20, help="The number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=100, help="The frequence to summarize and save model")
    parser.add_argument("--eval_freq", type=int, default=1000, help="The frequence to evaluate model")
    parser.add_argument("--save_latest_freq", type=int, default=5000, help="The frequence to save model")


    args = None
    try:
        args = parser.parse_args()
        print(args)
    except:
        return 1

    # Initialize trainer object
    m_trainer = RNN_depth_trainer()

    # Initialize data loading object
    dataLoader = m_trainer.initDataloader(args.dataset_dir, num_epochs=args.num_epochs)

    # A boolean evaluate every # steps
    eval_step = tf.placeholder(tf.bool, [])


    if args.eval_set_dir is not None:
        eval_dataLoader = m_trainer.initDataloader(args.eval_set_dir, num_epochs=None,is_training=False)


    # Multiple GPU
    devices = get_available_gpus()

    # Multiple grad and loss
    tower_grads = []
    tower_loss = []

    # Optimization method
    learning_rate = 0.0001
    beta = 0.9
    global_step = tf.train.get_or_create_global_step()
    optim = tf.train.AdamOptimizer(learning_rate, beta)

    controller="/cpu:0"
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)

            # Loop through all available GPU
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):

                # Load a sample

                data_dict = m_trainer.load_data(dataLoader)

                # If do evaluation, then load from evaluation dir
                if args.eval_set_dir is not None:
                    eval_data_dict = m_trainer.load_data(eval_dataLoader)
                    data_dict = tf.cond(eval_step, lambda:eval_data_dict, lambda:data_dict)

                # Forward network
                estimates = m_trainer.construct_model(data_dict) #

                # Compute Loss
                losses, all_losses, output_dict = m_trainer.compute_loss(estimates,  data_dict, global_step)#  est_depths_bw, est_poses_bw,

                tower_loss.append(losses)

                # Construct summary
                # Compute gradients
                with tf.name_scope("compute_gradients"):
                    # Get the gradient pairs (Tensor, Variable)
                    #depth_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pose_net")
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies([tf.group(*update_ops)]):
                        grads = optim.compute_gradients(losses)#,var_list=depth_vars)
                        tower_grads.append(grads)

            outer_scope.reuse_variables()

    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        apply_gradient_op = optim.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(tower_loss)

    m_trainer.tb_summary(output_dict, avg_loss,all_losses) #

    # Run
    m_trainer.train(apply_gradient_op, avg_loss, eval_step, args, data_dict)



if __name__ == "__main__":
    sys.exit(main())
