import argparse
import timeit
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as ski
from PIL import Image
from tqdm import tqdm
from scipy.spatial import ConvexHull
from skimage.color import rgb2gray
from skimage.measure import (find_contours, label, regionprops,
                             regionprops_table)

"""
This chunk defines functions necessary to calculate Ferret diameter using bounding area. 
Borrowed from https://bitbucket.org/william_rusnack/minimumboundingbox/src/master/
"""


def get_minFD(points):
    hull_ordered = [points[index] for index in ConvexHull(points).vertices]
    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)

    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered) - 1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    return min_rectangle['length_orthogonal']


def bounding_area(index, hull):
    unit_vector_p = unit_vector(hull[index], hull[index + 1])
    unit_vector_o = orthogonal_vector(unit_vector_p)

    dis_p = tuple(np.dot(unit_vector_p, pt) for pt in hull)
    dis_o = tuple(np.dot(unit_vector_o, pt) for pt in hull)

    min_p = min(dis_p)
    min_o = min(dis_o)
    len_orth = max(dis_o) - min_o
    len_p = max(dis_p) - min_p

    return {'area': len_p * len_orth, 'length_orthogonal': len_orth}


def unit_vector(pt0, pt1):
    # returns an unit vector that points in the direction of pt0 to pt1
    dis_0_to_1 = sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    # from vector returns a orthogonal/perpendicular vector of equal length
    return -1 * vector[1], vector[0]


""" <end borrowing> """


def create_heatmap(data, colormap) -> Image:
    """TODO Docstring"""

    # Create labeled fibers from mask:
    gray_obj = rgb2gray(data)
    bw_obj = gray_obj[:, :] == 1  # boolean matrix of borders
    bw_obj = np.array(bw_obj, dtype='int')  # change boolean to int
    labeled_image = label(bw_obj, background=1)  # label the objects using skimage

    # 'coords' are the y,x pixels that comprise the region
    properties = ['label', 'area', 'centroid', 'coords']

    print("Calculating fiber sizes...")
    tables = regionprops_table(labeled_image, properties=properties)
    tables = pd.DataFrame(tables)

    tables['minFD'] = tables['coords'].apply(get_minFD)  # Calculate minFD using get_minFD()

    # Convert things to int:
    tables = tables.astype({"centroid-0": int, "centroid-1": int})

    """ Created color-labeled image: """
    img_size = data.shape
    img = np.zeros(img_size[0:2], dtype='int')

    # assign minFD number to fiber coords:
    print("Iterating through fibers...")
    for i in tqdm(range(len(tables['coords']))):
        try:
            img[tables['coords'][i][:, 0], tables['coords'][i][:, 1]] = tables['minFD'][i]
        except:
            print(f"Assignment failed at index: {i}")
            continue

    """ Normalize image: """
    # TODO: fix this garbage...
    max_intensity = 200
    # max_intensity = np.amax(img)
    img = img / max_intensity
    cm = plt.get_cmap(colormap)
    heatmap_out = cm(img)[:, :, 0:3]
    heatmap_out = Image.fromarray((heatmap_out * 255).astype(np.uint8))
    heatmap_out = heatmap_out.convert('RGB')

    return heatmap_out


def init_argparse() -> argparse.ArgumentParser:
    parse = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Create minimum Feret diameter color-coded Fiber map."
    )
    parse.add_argument('-v', '--version', action='version', version=f"{parse.prog} version 1.0.0")
    parse.add_argument('-m', '--colormap', default='Purples', help='choose heatmap colormap')
    parse.add_argument('-c', '--csa', action='store_true', help='heatmap determined using minFD or CSA')
    parse.add_argument('-o', '--out_file_name', default='',
                       help='Output filename. \n Note: Exclude whitespace and include extensions')

    parse.add_argument('files', nargs=1)
    # parse.add_argument('files', nargs='*')    # TODO add multifile support

    return parse


if __name__ == "__main__":
    start_time = timeit.default_timer()
    parser = init_argparse()
    args = parser.parse_args()

    # TODO Add overwrite outfile check

    try:
        # Do operation on files...
        # for file in args.files: # TODO Add multi-file support later...

        print('Reading input file...')
        file_data = ski.imread(args.files[0])
        read_file_time = timeit.default_timer()
        print(f'Completed in {(read_file_time - start_time):.2f} seconds')
        print('Creating heatmap...')
        heatmap_image = create_heatmap(file_data, args.colormap)
        if args.out_file_name:
            file_name_out = args.out_file_name
        else:
            file_name_out = "output.tif"
        heatmap_time = timeit.default_timer()
        print(f'Completed in {(heatmap_time - read_file_time):.2f} seconds')
        print('Saving heatmap...')
        heatmap_image.save(file_name_out)
        save_time = timeit.default_timer()
        print(f'Completed in {(save_time - heatmap_time):.2f} seconds')

        [hm_w, hm_h] = heatmap_image.size
        [fd_h, fd_w, _] = file_data.shape

        assert hm_w == fd_w and hm_h == fd_h, "Error: heatmap and input file have different dimensions"  # TODO fix this

    except FileNotFoundError:
        print("ERROR: File not found. Did you specify a space-delimited list of images?")
    except:
        print("ERROR: Unknown error occurred. Use --verbose to print full error message")
        raise

    print('Success.')
