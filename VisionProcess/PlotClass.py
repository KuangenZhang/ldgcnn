# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:59:08 2019

@author: kuangen
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
class PlotClass:
    
    def plot_points_with_color(points,label,face_id, fig, axis_off = False):
        ax = fig.add_subplot(111, projection='3d')
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(face_id))
        ax.scatter(points[:,0], points[:,1], points[:,2], c = colors)
        ax.view_init(elev=0, azim=90)
        fontsize = 20
        if axis_off:
            ax.set_axis_off()
        else:
            ax.set_xlabel('x', fontsize=fontsize)
            ax.set_ylabel('y', fontsize=fontsize)
            ax.set_title(label, fontsize=fontsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1,wspace = 0, hspace = 0)
        plt.show()
    
    def plot_points(points,label, fig, axis_off = False):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2])
        ax.view_init(elev=0, azim=90)
        fontsize = 20
        if axis_off:
            ax.set_axis_off()
        else:
            ax.set_xlabel('x', fontsize=fontsize)
            ax.set_ylabel('y', fontsize=fontsize)
        ax.set_title(label, fontsize=fontsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1,wspace = 0, hspace = 0)
        plt.show()
    
    def subplot_color_points(points,label,colors, fig,nrows, ncols, index, 
                       axis_off = False):
        ax = fig.add_subplot(nrows, ncols, index, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], c = colors)
        ax.view_init(elev=45, azim=-45)
        fontsize = 20
        if axis_off:
            ax.set_axis_off()
        else:
            ax.set_xlabel('x', fontsize=fontsize)
            ax.set_ylabel('y', fontsize=fontsize)
            ax.set_title(label, fontsize=fontsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1,wspace = 0, hspace = 0)
    
    def subplot_points(points,label,fig,nrows, ncols, index, 
                       axis_off = False):
        ax = fig.add_subplot(nrows, ncols, index, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2])
        ax.view_init(elev=0, azim=0)
        fontsize = 20
        if axis_off:
            ax.set_axis_off()
        else:
            ax.set_xlabel('x', fontsize=fontsize)
            ax.set_ylabel('y', fontsize=fontsize)
            ax.set_title(label, fontsize=fontsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1,wspace = 0, hspace = 0)
        plt.show()
    
    def view_2D_mat(data,label = ''):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(data, interpolation='nearest')
        plt.axis('off')
        plt.show()
        plt.title(label)
        
    def sub_view_2D_mat(data,label,fig, nrows, ncols, index):
        ax = fig.add_subplot(nrows, ncols, index)
        ax.imshow(data, interpolation='nearest',cmap=plt.get_cmap('hot'),vmin=0, vmax=1)
        plt.axis('off')
        plt.show()
        plt.title(label)
        
    def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
    
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
        fig = plt.figure(figsize= (14,10.8))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
    #    fmt = '.2f' if normalize else 'd'
    #    thresh = cm.max() / 2.
    #    for i in range(cm.shape[0]):
    #        for j in range(cm.shape[1]):
    #            ax.text(j, i, format(cm[i, j], fmt),
    #                    ha="center", va="center",
    #                    color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return cm,ax