import os
import sys
import argparse

import numpy as np
from imageio import imwrite

from load_kinect_rgbd_images import get_color_images

def convert_color_images_to_png(ARGS):
    color_images = get_color_images(ARGS.file_color_pickle)
    color_images = np.array(color_images)
    num_color_images = color_images.shape[0]

    print(f"Num color images to convert: {num_color_images}")

    if not os.path.isdir(ARGS.dir_out_color_png_data):
        os.makedirs(ARGS.dir_out_color_png_data)

    for file_index in range(num_color_images):
        current_color_image = color_images[file_index]
        file_path_color_png = os.path.join(
            ARGS.dir_out_color_png_data,
            f"{ARGS.str_prefix_file_name}-{file_index:02d}.png",
        )
        print(file_path_color_png)
        imwrite(file_path_color_png, current_color_image)

    print(f"color images saved in: {ARGS.dir_out_color_png_data}")
    return

def main():
    parser = argparse.ArgumentParser(description="convert format of color images")
    parser.add_argument("--file_color_pickle", required=True,
        help="full path to color pickle file")
    parser.add_argument("--dir_out_color_png_data", required=True,
        help="full directory path where output color images in .png format need to be saved")
    parser.add_argument("--str_prefix_file_name", required=True,
        help="string prefix for file name")
    ARGS = parser.parse_args()
    convert_color_images_to_png(ARGS)
    return

if __name__ == "__main__":
    main()
