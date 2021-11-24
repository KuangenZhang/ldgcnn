# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:39:41 2019

@author: kuangen
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../VisionProcess'))
from PlotClass import PlotClass
from FileIO import FileIO
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
#%%

def plot_segmentation_results():
    obj_file = '_pred'
    #obj_file = '_gt'
    fig_size = [2, 8] #width,height
    fig = plt.figure(figsize= (fig_size[0],fig_size[1]))
    idx_vec = np.array([45,163,263,13,16]) # plane, motorbike, car, chair, table
    for r in range(5):
        file_name = 'test_results/' + str(idx_vec[r]) + obj_file + '.obj'
        cloud, colors = FileIO.load_obj_file(file_name)
        np.save('images/{}.npy'.format(idx_vec[r]), cloud)
        print(cloud.shape)
        # switch z and y axis
        y = np.copy(cloud[:, 1])
        cloud[:, 1] = cloud[:, 2]
        cloud[:, 2] = y
        PlotClass.subplot_color_points (cloud,'', colors ,fig, 5, 1, r+1,
                       axis_off = True)
    time_str = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    bbox = fig.bbox_inches.from_bounds(0, 0, fig_size[0], fig_size[1])#for single plot
    img_dir = 'images'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.savefig(img_dir +'/'+ time_str + obj_file + '_test.png',bbox_inches=bbox)
    plt.show()

def visualize_feature_distance(obj_idx = 45):
    features = np.load('visualization_results/{}_features.npy'.format(obj_idx), allow_pickle=True).item()
    pts = features['point_cloud'].squeeze()

    center_idx_list = [np.argmax(pts[:, 0]), np.argmax(pts[:, 1]), np.argmax(pts[:, 2])]
    feature_distance_list = []
    key_list = ['point_cloud', 'net1', 'net2', 'net3', 'net4']
    for r in range(len(center_idx_list)):
        feature_distances = []
        for c in range(len(key_list)):
            feature = features[key_list[c]].squeeze()
            feature_distance = calc_feature_distance(feature, center_idx_list[r])
            feature_distances.append(feature_distance)
        feature_distance_list.append(feature_distances)
    plot_point_cloud_rendered_by_feature_distance(pts, feature_distance_list, center_idx_list)

def calc_feature_distance(feature, center_idx):
    '''
    :param feature: [num_points * feature_channel]
    :param center_idx: idx of the center point
    :return: feature distance from each point to the center point
    '''
    feature_distance = np.sum((feature - feature[[center_idx]])**2, axis = -1)
    return feature_distance

def plot_point_cloud_rendered_by_feature_distance(pts, feature_distance_list, center_idx_list):
    fig = plt.figure(figsize= (16, 9))
    rows, cols = len(center_idx_list), len(feature_distance_list[0])
    xs, ys, zs = pts[:, 0], pts[:, 2], pts[:, 1]
    cm = plt.get_cmap("winter")
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, r * cols + c + 1, projection='3d')
            feature_distance = feature_distance_list[r][c] # the feature distance for center idx and net c
            center_idx = center_idx_list[r]
            ax.scatter(xs, ys, zs, c=-feature_distance, cmap = cm)
            ax.scatter(xs[center_idx], ys[center_idx], zs[center_idx], c='r', s=50)
            ax.axis('off')
    plt.show()

if __name__=='__main__':
  # plot_segmentation_results()
  visualize_feature_distance()
