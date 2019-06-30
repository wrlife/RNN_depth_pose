import os
import sys
import argparse
from multiprocessing import Pool
from kitti_utils import *


def main():

    parser = argparse.ArgumentParser(description="Generates the sun3d training datasets.")
    parser.add_argument("--kitti_path", type=str, required=True, help="The path to the kitti data directory")
    parser.add_argument("--outputdir", type=str, default='training_data', help="Output directory for the generated "
                                                                               "tfrecords files")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads")

    args = None
    try:
        args = parser.parse_args()
        print(args)
    except:
        return 1

    kitti_path = args.kitti_path
    outputdir = args.outputdir
    os.makedirs(outputdir, exist_ok=True)
    threads = args.threads

    # read txt file with the train sequence names
    with open('kitti_test.txt', 'r') as f:
        sequences = f.read().splitlines()

    depth_path = '/playpen1/datasets/KITTI/KITTI_depth/train/'
    #with Pool(threads) as pool:
        # create temporary h5 files for each baseline and sequence combination
        # baseline_range_files_dict = {b:[] for b in baseline_ranges}
    args = []

    for i, seq_name in enumerate(sequences):
        print(seq_name)
        outfile = os.path.join(outputdir, 'cam3_'+ seq_name + ".tfrecords")
        args.append((outfile, kitti_path,depth_path, seq_name))

        #created_groups = pool.starmap(Rui_create_samples_from_sequence_kitti, args, chunksize=1)

        create_samples_from_sequence_kitti(outfile, kitti_path, depth_path, seq_name)

    print('created', sum(created_groups), 'groups')

    return 0


if __name__ == "__main__":
    sys.exit(main())
