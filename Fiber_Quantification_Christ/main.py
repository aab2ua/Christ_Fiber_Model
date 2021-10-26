# Using argparser for the fiber model python notebook
# include necessary packages
import argparse
from scipy.spatial import ConvexHull
from math import sqrt
import skimage.io as ski
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
from skimage.measure import label, regionprops, find_contours
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from PIL import Image
from sklearn.cluster import KMeans
import cv2  # this is the main openCV class, the python binding file should be in /pythonXX/Lib/site-packages


# construct argument parse
def init_argparse():
    parse = argparse.ArgumentParser()
    #ap.add_argument("SMASH_mask", type = str, action = 'store', help = "Input the filename of the SMASH mask you are analyzing")
    parse.add_argument('files', nargs = 1)
    parse.add_argument('ratio', nargs = 1)
    #args = ap.parse_args()
    return parse

# defined functions in script

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

    return {'area': len_p * len_orth,
            'length_orthogonal': len_orth,
            }

def unit_vector(pt0, pt1):
    # returns an unit vector that points in the direction of pt0 to pt1
    dis_0_to_1 = sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    # from vector returns a orthogonal/perpendicular vector of equal length
    return -1 * vector[1], vector[0]

def labeled_fibers(fiber_masks):
    # Create labeled fibers from mask
    gray_obj = rgb2gray(fiber_masks)
    bw_obj = gray_obj[:, :] == 1
    bw_obj = np.array(bw_obj, dtype='int')
    labeled_image = label(bw_obj, background=1)  # Make new image with regions colored corresponding to minFD:

    # Determine properties from label:
    properties = ['label', 'area', 'centroid', 'coords']

    tables = regionprops_table(labeled_image, properties=properties)
    tables = pd.DataFrame(tables)

    tables = tables[tables['area'] > 10]
    tables['minFD'] = tables['coords'].apply(get_minFD)

    # Convert centroid to int to line up with pixels:
    tables = tables.astype({"centroid-0": int, "centroid-1": int})
    return tables

def heatmap(fiber_masks, tables):
    img_size = fiber_masks.shape
    img = np.zeros(img_size[0:2], dtype='int')

    for i in range(len(tables['coords'])):
        try:
            img[tables['coords'][i][:, 0], tables['coords'][i][:, 1]] = tables['minFD'][i]
        except:
            print(f"Assignment failed at index: {i}")
            continue

    # Normalize:
    max_intensity = np.amax(img)
    img = img / max_intensity

    # Apply colormap
    cm = plt.get_cmap("terrain")
    img_out = cm(img)[:, :, 0:3]

    img_out = Image.fromarray((img_out * 255).astype(np.uint8))
    img_out = img_out.convert('RGB')
    return img_out

def kmeans(tables, heatmap):
    print('a')
    small_fibers = tables[tables['minFD'] < 25]
    print('b')
    centroids = small_fibers[['centroid-1', 'centroid-0']].to_numpy()
    print('f')
    kmeans = KMeans(n_clusters=2, random_state=0).fit(centroids)
    print('c')
    # Plot cluster regions:
    [width, height] = heatmap.size
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print('d')
    xcoord = kmeans.cluster_centers_[:, 0][0]
    ycoord = kmeans.cluster_centers_[:, 1][0]
    print('e')
    centroid = [xcoord, ycoord]
    return centroid

def contouring(mask):
    # Grayscaling image
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # change to grayscale
    gray = gray.astype(np.uint8)

    # Thresholding image
    ret,thresh = cv2.threshold(gray,254,255,cv2.THRESH_BINARY) #100 gives a good image in the below chunk

    # Inverting colors
    inverted = cv2.bitwise_not(thresh)

    # Blurring image
    blurred = cv2.GaussianBlur(inverted, (151, 151), 0)

    # Drawing contours
    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # find contours with simple approximation
    cv2.drawContours(mask, contours, -1, (80, 40, 70), 50)
    return mask, contours

def small_contour_point(x1, y1, x2, y2, ratio):
    ratio = ratio / 100
    small_contour_x = ((x2 - x1) * ratio) + x1
    small_contour_y = ((y2 - y1) * ratio) + y1
    return (small_contour_x, small_contour_y)

def smaller_contour(contours, ratio, xcoord, ycoord, fiber_mask):
    small_contour = []
    for x in range(0, len(contours) - 1):
        array = contours[x]
        for y in range(0, len(array) - 1):
            point = array[y]
            point = point[0]
            contour_x = point[0]
            contour_y = point[1]
            new_point = small_contour_point(int(xcoord), int(ycoord), contour_x, contour_y,
                                            ratio)  # xcoord and ycoord is centroid coordinates from kmeans
            small_contour.append(new_point)
            cv2.circle(fiber_mask, (int(new_point[0]), int(new_point[1])), 50, [255, 255, 255],
                       -1)  # Color the outline white so that it's creating "two sections"
    return fiber_mask

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    try:
        print('Reading input file...')
        fiber_masks = ski.imread(args.files[0])
        print('Creating heatmap...')
        tables = labeled_fibers(fiber_masks)
        heatmap_image = heatmap(fiber_masks, tables)
        heatmap_image.save("output.tif")
        print('Calculating centroid with kmeans')
        centroid = kmeans(tables, heatmap_image)
        print('Contouring mask')
        con_mask, contours = contouring(fiber_masks)
        print('Creating smaller contour')
        ratio = args.files[1]
        defined_mask = smaller_contour(contours, ratio, centroid[0], centroid[1], con_mask)
        print("Success")
    except:
        print("Something went wrong \(' ')/")








