# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:39:41 2019

@author: kuangen
"""
import sys
import os
import json
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../VisionProcess'))
from PlotClass import PlotClass
from FileIO import FileIO
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib import colors
#%%
hdf5_data_dir = os.path.join(BASE_DIR, 'hdf5_data')
color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))
my_cmp = colors.LinearSegmentedColormap.from_list('my_colormap', color_map)

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
    seg_pred = features['seg_pred'].squeeze()
    point_num = len(seg_pred)
    pts = features['point_cloud'].squeeze()[:point_num]
    print(seg_pred)
    idx_list = np.arange(point_num)
    # center_idx_list = [99, idx_list[seg_pred==2][0], idx_list[seg_pred==3][-1]] # plane
    seg_pred_unique = np.unique(seg_pred)
    center_idx_list = [idx_list[seg_pred==seg_pred_unique[0]][0],
                       idx_list[seg_pred==seg_pred_unique[1]][0],
                       idx_list[seg_pred==seg_pred_unique[2]][-1]] # chair
    feature_distance_list = []
    key_list = ['point_cloud', 'net1', 'net2', 'net3', 'net4']
    for r in range(len(center_idx_list)):
        feature_distances = []
        for c in range(len(key_list)):
            feature = features[key_list[c]].squeeze()[:point_num]
            feature_distance = calc_feature_distance(feature, center_idx_list[r])
            feature_distances.append(feature_distance)
        feature_distance_list.append(feature_distances)
    plot_point_cloud_rendered_by_feature_distance(pts, seg_pred, feature_distance_list, center_idx_list)
    # plt.savefig('images/feature_{}.png'.format(obj_idx), dpi=300)
    plt.show()


def save_color_bar():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    data = np.random.rand(11, 16)
    im = ax.imshow(data, cmap=plt.get_cmap("winter"))
    cbar = fig.colorbar(im, orientation="horizontal", pad=0.2)
    cbar.set_ticks([np.min(data), np.max(data)])
    cbar.set_ticklabels(["Near", "Far"])
    plt.savefig('images/colorbar.pdf', dpi=300)
    plt.show()


def calc_feature_distance(feature, center_idx):
    '''
    :param feature: [num_points * feature_channel]
    :param center_idx: idx of the center point
    :return: feature distance from each point to the center point
    '''
    feature_distance = np.sqrt(np.sum((feature - feature[[center_idx]])**2, axis = -1))
    return feature_distance

def plot_point_cloud_rendered_by_feature_distance(pts, seg_pred, feature_distance_list, center_idx_list):
    fig = plt.figure(figsize= (16, 6))
    rows, cols = len(center_idx_list), len(feature_distance_list[0])
    xs, ys, zs = pts[:, 0], pts[:, 2], pts[:, 1]
    cm = plt.get_cmap("winter")
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols+1, r * (cols+1) + c + 1, projection='3d')
            feature_distance = feature_distance_list[r][c] # the feature distance for center idx and net c
            center_idx = center_idx_list[r]
            ax.scatter(xs, ys, zs, c=feature_distance, cmap = cm, s=10)
            ax.scatter(xs[center_idx]-0.1, ys[center_idx], zs[center_idx]+0.1, c='r', s=20)
            # ax.axis('off')
            plt.xlabel('x')
            plt.ylabel('y')
            azm = ax.azim
            ele = ax.elev
            ax.view_init(elev=ele, azim=azm+180)# lamp
            # ax.view_init(elev=60, azim=0)# plane
    for r in range(rows):
        ax = fig.add_subplot(rows, cols + 1, (r+1) * (cols + 1), projection='3d')
        ax.scatter(xs, ys, zs, c=seg_pred, cmap=my_cmp, s=10)
        ax.axis('off')
        ax.view_init(elev=0, azim=0)  # lamp
        # ax.view_init(elev=ele, azim=azm+45) # chair
        # ax.view_init(elev=60, azim=0) # plane
    plt.tight_layout()

def visualize_segmentation():
    fig = plt.figure(figsize=(4, 3))
    for i in range(0, 500, 5):
        features = np.load('visualization_results/{}_features.npy'.format(i), allow_pickle=True).item()
        seg_pred = features['seg_pred'].squeeze()
        point_num = len(seg_pred)
        pts = features['point_cloud'].squeeze()[:point_num]
        xs, ys, zs = pts[:, 0], pts[:, 2], pts[:, 1]
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, c=seg_pred, cmap=my_cmp, s=10)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('images/segmentation_results/{0:03d}.png'.format(i), dpi=300)


def plot_all_segmentation():
    fig_size = [4, 3]
    fig = plt.figure(figsize=tuple(fig_size))
    for i in range(0, 2800, 10):
        file_name = 'test_results/' + str(i) + '_pred.obj'
        cloud, colors = FileIO.load_obj_file(file_name)
        y = np.copy(cloud[:, 1])
        cloud[:, 1] = cloud[:, 2]
        cloud[:, 2] = y
        PlotClass.subplot_color_points(cloud, '', colors, fig, 1, 1, 1,
                                       axis_off=True)
        bbox = fig.bbox_inches.from_bounds(0, 0, fig_size[0], fig_size[1])  # for single plot
        img_dir = 'images/all_segmentation'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        plt.savefig(img_dir + '/{0:4d}.png'.format(i), bbox_inches=bbox, dpi = 300)

if __name__=='__main__':
  # plot_segmentation_results()
  # visualize_feature_distance(obj_idx=390)
  # save_color_bar()
  plot_all_segmentation()
