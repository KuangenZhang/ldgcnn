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
sys.path.append(os.path.join(BASE_DIR, '../../../VisionProcess'))
from PlotClass import PlotClass
from PclAlgo import PclAlgo
from FileIO import FileIO
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
#%%
obj_file = '_pred'
#obj_file = '_gt'
fig_size = [2, 8] #width,height
fig = plt.figure(figsize= (fig_size[0],fig_size[1]))
idx_vec = np.array([45,163,263,13,16])
#np.random.shuffle(idx_vec)
for r in range(6):
    file_name = 'test_results/' + str(idx_vec[r]) + obj_file + '.obj'
    cloud, colors = FileIO.load_obj_file(file_name)
    cloud = PclAlgo.switch_z_y(cloud)
    PlotClass.subplot_color_points (cloud,'', colors ,fig, 5, 1, r+1, 
                   axis_off = True)
time_str = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
bbox = fig.bbox_inches.from_bounds(0, 0, fig_size[0], fig_size[1])#for single plot
plt.savefig('images/'+ time_str + obj_file + '_test.png',bbox_inches=bbox)