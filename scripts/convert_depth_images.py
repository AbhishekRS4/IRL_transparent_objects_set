import os
import sys
import argparse

import numpy as np

from exr_utils import exr_saver
from load_kinect_rgbd_images import get_depth_images

def convert_depth_images_to_exr(ARGS):
    EXT_DEPTH_IMG = "-transparent-depth-img.exr"

    depth_images = get_depth_images(ARGS.file_depth_pickle)
    depth_images = np.array(depth_images)
    num_depth_images = depth_images.shape[0]

    print(f"Num depth images to convert: {num_depth_images}")

    if not os.path.isdir(ARGS.dir_out_depth_exr_data):
        os.makedirs(ARGS.dir_out_depth_exr_data)

    for file_index in range(num_depth_images):
        current_depth_image = depth_images[file_index]
        # current_depth_image = current_depth_image.astype(np.int16)<<2
        current_depth_image = current_depth_image.astype(np.float32)
        current_depth_image /= 1000.
        # convert mm to metres
        file_path_depth_exr = os.path.join(
            ARGS.dir_out_depth_exr_data,
            f"{ARGS.str_prefix_file_name}-{file_index:02d}{EXT_DEPTH_IMG}",
        )
        print(file_path_depth_exr)
        exr_saver(file_path_depth_exr, current_depth_image, ndim=1)

    print(f"depth images saved in: {ARGS.dir_out_depth_exr_data}")
    return

def main():
    parser = argparse.ArgumentParser(description="convert format of depth images")
    parser.add_argument("--file_depth_pickle", required=True,
        help="full path to depth pickle file")
    parser.add_argument("--dir_out_depth_exr_data", required=True,
        help="full directory path where output depth images in .exr format need to be saved")
    parser.add_argument("--str_prefix_file_name", required=True,
        help="string prefix for file name")
    ARGS = parser.parse_args()
    convert_depth_images_to_exr(ARGS)
    return

if __name__ == "__main__":
    main()
