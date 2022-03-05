# Using argparser for the fiber model python notebook
# include necessary packages
from __future__ import generators

import argparse
from scipy.spatial import ConvexHull
from scipy import sparse
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
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill
import math
from math import sqrt

# construct argument parse
def init_argparse():
    parse = argparse.ArgumentParser()
    #ap.add_argument("SMASH_mask", type = str, action = 'store', help = "Input the filename of the SMASH mask you are analyzing")
    parse.add_argument('files', nargs = 1)
    parse.add_argument('ratio', nargs = 1)
    #args = ap.parse_args()
    return parse

# defined functions in script
def min_max_feret(Points):
    '''Given a list of 2d points, returns the minimum and maximum feret diameters.'''
    squared_distance_per_pair = [((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, (p, q))
                                 for p, q in rotatingCalipers(Points)]
    min_feret_sq, min_feret_pair = min(squared_distance_per_pair)
    max_feret_sq, max_feret_pair = max(squared_distance_per_pair)
    return sqrt(min_feret_sq), sqrt(max_feret_sq)

# { functions for min_max_feret
def rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    U, L = hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i], L[j]

        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1:
            j -= 1
        elif j == 0:
            i += 1

        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i + 1][1] - U[i][1]) * (L[j][0] - L[j - 1][0]) > \
                (L[j][1] - L[j - 1][1]) * (U[i + 1][0] - U[i][0]):
            i += 1
        else:
            j -= 1


def orientation(p, q, r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1] - p[1]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[1] - p[1])


def hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2], U[-1], p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2], L[-1], p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U, L
# }

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

# { functions for get_minFD
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
# }

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

def cent_mode(tables): #either run this or kmeans function to find centroid
    small_fibers = tables[tables['minFD'] < 25]

    # 'centroid-1' is x coordinates and 'centroid-0' is y coordinates
    # finding x centroid coord
    small_centroids_x = small_fibers['centroid-1'].mode()
    tot_cent_x = tables['centroid-1'].mean()

        # going through the x coord modes and picking the smallest one
    small_cent_x = small_centroids_x[0].item()
    old_dist = math.dist([small_cent_x], [tot_cent_x])

    for i in range(0, len(small_fibers['centroid-1'].mode())):
        if i != 0:
            old_dist = dist
        trial_cent_x = small_centroids_x[i].item()
        dist = math.dist([trial_cent_x], [tot_cent_x])
        if dist < old_dist:
            small_cent_x = trial_cent_x
    # find y centroid coord
    small_centroids_y = small_fibers['centroid-0'].mode()
    tot_cent_y = tables['centroid-0'].mean()

        # going through the x coord modes and picking the smallest one
    small_cent_y = small_centroids_y[0].item()
    old_dist = math.dist([small_cent_y], [tot_cent_y])

    for i in range(0, len(small_fibers['centroid-0'].mode())):
        if i != 0:
            old_dist = dist
        trial_cent_y = small_centroids_y[i].item()
        dist = math.dist([trial_cent_y], [tot_cent_y])
        if dist < old_dist:
            small_cent_y = trial_cent_y

    return small_cent_x, small_cent_y, small_fibers



def kmeans1(tables, heatmap):
    small_fibers = tables[tables['minFD'] < 25]
    centroids = small_fibers[['centroid-1', 'centroid-0']].to_numpy()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(centroids)
    [width, height] = heatmap.size
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    xcoord = kmeans.cluster_centers_[:, 0][0]
    ycoord = kmeans.cluster_centers_[:, 1][0]
    centroid = [xcoord, ycoord]
    return centroid, small_fibers

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

def total_contours(contours):
    total_contours = []

    for x in range(0, len(contours)):
        section = contours[x]
        sec_len = len(section)

        for z in range(0, sec_len):
            section2 = section[z]
            while section2.ndim != 1:
                section2 = section2[0]

            point = (section2[0], section2[1])
            total_contours.append(point)

    return total_contours

def smaller_contour(contours, ratio, xcoord, ycoord, fiber_mask):
    small_contour = []
    contours = total_contours(contours)

    for x in range(0, len(contours) - 1):
        point = contours[x]
        contour_x = point[0]
        contour_y = point[1]
        new_point = small_contour_point(int(xcoord), int(ycoord), contour_x, contour_y,
                                        ratio)  # xcoord and ycoord is centroid coordinates from kmeans
        small_contour.append(new_point)
        cv2.circle(fiber_mask, (int(new_point[0]), int(new_point[1])), 50, [255, 255, 255], -1)  # Color the outline white so that it's creating "two sections"

    return fiber_mask, small_contour

def binary_filtering(def_mask, fiber_masks, small_contour):
    # 1. Creating outline
    x = def_mask.shape[0]
    y = def_mask.shape[1]
    z = def_mask.shape[2]
    mask = np.zeros((x,y,z), dtype = np.uint8)
    for x in range(0, len(small_contour) - 1):
        cv2.circle(mask, (int(small_contour[x][0]), int(small_contour[x][1])), 50, [255, 255, 255], -1)  # Color the outline white so that it's creating "two sections"

    # 2. Filling in outline
    filled_mask = flood_fill(mask, (2000, 3500, 0), 255)  # need to mke this automated
    if filled_mask[0, 0][
        0] == 255:  # Sometimes the mask inverts so this make sure the mask has a black background      #255 means white
        filled_mask = cv2.bitwise_not(filled_mask)

    # 3. Grayscaling original mask
    for y in range(0, fiber_masks.shape[1]):  # https://scikit-image.org/docs/dev/user_guide/numpy_images.html
        for x in range(0, fiber_masks.shape[0]):
            if np.any(fiber_masks[x, y] != 255):
                fiber_masks[x, y] = 50;
    print('f')
    filtered_mask = fiber_masks * filled_mask

    return filled_mask, filtered_mask

def fiber_count(filtered_mask, tables):
    fiber_count = 0
    core_index = []
    outer_index = []

    for i in range(len(tables)):
        if np.any(filtered_mask[tables['centroid-0'][i], tables['centroid-1'][i]] != 0):
            fiber_count += 1
            core_index.append(i)
        else:
            outer_index.append(i)

    return fiber_count, core_index, outer_index

def area_stats(core_index, outer_index, tables):
    #Separating tables into core and outer regions
    core_region_tables = pd.DataFrame()
    for i in range(len(core_index)):
        j = core_index[i]
        row = {"area": tables.loc[j][1], "original index": j, "minFD": tables.loc[j][5]}
        core_region_tables = core_region_tables.append(row, ignore_index=True)

    outer_region_tables = pd.DataFrame()
    for i in range(len(outer_index)):
        j = outer_index[i]
        row = {"area": tables.loc[j][1], "original index": j,
               "minFD": tables.loc[j][5]}  # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
        outer_region_tables = outer_region_tables.append(row, ignore_index=True)  # https://www.stackvidhya.com/add-row-to-dataframe/

    # Building histograms
    outer_area = outer_region_tables['area']
    core_area = core_region_tables['area']

    area_hist = plt.figure(figsize=(8, 6))
    plt.hist(outer_area, bins=100, alpha=0.5, label="Outer region area")
    plt.hist(core_area, bins=100, alpha=0.5, label="Core region area")
    plt.xlabel("Fiber Area", size=14)
    plt.ylabel("Frequency", size=14)
    plt.title("Outer vs. Core Histograms for Area")
    plt.legend(loc='upper right')

    # Building csvs
    area_summary = pd.merge(outer_area.describe(), core_area.describe(),
                            left_index=True, right_index=True,
                            suffixes=('_outer', '_core'))

    area_raw = pd.merge(outer_area, core_area, left_index=True,
                        right_index=True, suffixes=('_outer', '_core'))

    return area_hist, area_summary, area_raw

def minFD_stats(core_index, outer_index, tables):
    #Separating tables into core and outer regions
    core_region_tables = pd.DataFrame()
    for i in range(len(core_index)):
        j = core_index[i]
        row = {"area": tables.loc[j][1], "original index": j, "minFD": tables.loc[j][5]}
        core_region_tables = core_region_tables.append(row, ignore_index=True)

    outer_region_tables = pd.DataFrame()
    for i in range(len(outer_index)):
        j = outer_index[i]
        row = {"area": tables.loc[j][1], "original index": j,
               "minFD": tables.loc[j][5]}  # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
        outer_region_tables = outer_region_tables.append(row,
                                                         ignore_index=True)  # https://www.stackvidhya.com/add-row-to-dataframe/

    # Building Stats:
    outer_minFD = outer_region_tables['minFD']
    core_minFD = core_region_tables['minFD']

    minFD_hist = plt.figure(figsize=(8, 6))
    plt.hist(outer_minFD, bins=100, alpha=0.5, label="Outer region minFD")
    plt.hist(core_minFD, bins=100, alpha=0.5, label="Core region minFD")
    plt.xlabel("Feret Diameter", size=14)
    plt.ylabel("Frequency", size=14)
    plt.title("Outer vs. Core Histograms for MinFD")
    plt.legend(loc='upper right')

    # Building csvs:
    minFD_summary = pd.merge(outer_minFD.describe(), core_minFD.describe(),
                             left_index=True, right_index=True,
                             suffixes=('_outer', '_core'))

    minFD_raw = pd.merge(outer_minFD, core_minFD, left_index=True,
                         right_index=True, suffixes=('_outer', '_core'))

    return minFD_hist, minFD_summary, minFD_raw

def edited_heatmap(heatmap, small_contour):
    edit_heatmap = np.array(heatmap)

    for x in range(0, len(small_contour) - 1):
        cv2.circle(edit_heatmap, (int(small_contour[x][0]), int(small_contour[x][1])), 50, [255, 255, 255],
                   -1)  # Color the outline white so that it's creating "two sections"
    return edit_heatmap

def plotting_modes(heatmap, x):
    heatmap = np.array(heatmap)
    cv2.circle(heatmap, (x, 0), 50, [255, 0, 0], -1)
    return heatmap

def finding_mode_x(small_fibers, heatmap):
    mode_x = small_fibers['centroid-1'].mode()
    if len(mode_x) == 1:
        trial_mode = int(mode_x)
    print(trial_mode)

    mode = plotting_modes(heatmap, mode_x)
    mode = Image.fromarray(mode)
    mode.save("first mode.tif")

    keep_mode = input("Is this mode in the right cluster? Y/N: ")
    delete_mode = False

    if keep_mode == "N":
        delete_mode = True
        small_fibers.drop(small_fibers[small_fibers['centroid-1'] == mode_x].index, inplace = True)
        mode_x = small_fibers['centroid-1'].mode()
        print(mode_x)

    return mode_x




#________________________________________________________________________-
if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    try:
        print('Reading input file...')
        fiber_masks = ski.imread(args.files[0])
        ratio = int(args.ratio[0])
        print('Creating heatmap...')
        tables = labeled_fibers(fiber_masks)
        heatmap_image = heatmap(fiber_masks, tables)
        print('Calculating centroid with mode')
        centroid, small_fibers = kmeans1(tables, heatmap_image)
        #centroid = cent_mode(tables)
        # print('Calculating minimum and maximum Feret diameters')
        # #min_FD, max_FD = min_max_feret()
        # #figure out what the points are to input into function
        # print('Contouring mask')
        # #os.makedirs(fiber_masks + ": " + str(ratio) + "% analysis")
        # #os.chdir(fiber_masks + ": " + str(ratio) + "% analysis")
        # con_mask, contours = contouring(fiber_masks)
        # contoured_mask = Image.fromarray(con_mask)
        # contoured_mask.save("con_mask.jpeg")
        # print('Creating smaller contour')
        # def_mask, small_contour = smaller_contour(contours, ratio, centroid[0], centroid[1], fiber_masks)
        # defined_mask = Image.fromarray(def_mask)
        # defined_mask.save("def_mask.jpeg")
        # print('Binary Filtering')
        # filled_mask, filtered_mask = binary_filtering(def_mask, fiber_masks, small_contour)
        # print('Fiber Count')
        # fiber_count, core_index, outer_index = fiber_count(filtered_mask, tables)
        # print('Area Data Analysis')
        # area_hist, area_summary, area_raw = area_stats(core_index, outer_index, tables)
        # area_hist.savefig("Outer_vs_Core_Area.png")
        # area_summary.to_csv("Area_Summary.csv")
        # area_raw.to_csv("Area_Raw.csv")
        # print('MinFD Data Analysis')
        # minFD_hist, minFD_summary, minFD_raw = minFD_stats(core_index, outer_index, tables)
        # minFD_hist.savefig("Outer_vs_Core_minFD.png")
        # minFD_summary.to_csv("minFD_Summary.csv")
        # minFD_raw.to_csv("minFD_Raw.csv")
        #print('Saving edited heatmap image')
        #new_heatmap = edited_heatmap(heatmap_image, small_contour)
        #heatmap = Image.fromarray(new_heatmap)
        #heatmap.save("heatmap.tif")
        print("trying out modes")
        mode_x = finding_mode_x(small_fibers, heatmap_image)
        print("Success")
    except FileNotFoundError:
        print("Can't find the right file")
    except:
        print("Something went wrong")




