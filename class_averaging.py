# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:10:04 2020

@author: Janak MIT
"""
"""
This file uses text files in Data folder outputted by vaecluster_loop and summarizes them. 

"""
import numpy as np
import os
import matplotlib.pyplot as plt



### Selecting which compositions to use, max 44 selects all the way from 99 to min 56, works without error from 1-43, need to check 44
sel_range = []
sel_k = 0
ul_ = 20 #consistent with vaecluster_loop variable
for i in range(0,8):
    for j in range(0,ul_):
        sel_range.append(sel_k)
        sel_k += 1
    sel_k += 44-ul_

dim_text = '2d' #change if the latent space was not 2d(imensional)
tem_par = np.loadtxt('VO2 - Nb2O3 Composition and temp Combiview.txt', skiprows=1)[sel_range]

### Ensuring folder to save output exists
if not os.path.exists('Summary'):
    os.makedirs('Summary')
subpath = os.getcwd() +"\\Summary\\Groups"
if not os.path.exists(subpath):
    os.makedirs(subpath)


### A function to label changes in cluster assignment for data not near the edges
def labeling_rule(colbox, cluster_data_):
    indx_low = 4
    indx_high = ul_-1
    tru_list = []
    while indx_high < cluster_data_:
        for indx in range(indx_low, indx_high):
            if colbox[indx] != colbox[indx+1]:
                if colbox[indx] == colbox[indx-1] and colbox[indx]==colbox[indx-2]: 
                    tru_list.append(indx+1)
        indx_low += ul_
        indx_high += ul_
    return tru_list

### Defining variables, plotting parameters and saving outputs

temp = ["C"]
temp.extend(tem_par[:,0])
comp = ["T"]
comp.extend(tem_par[:,1])

figdpi = 600
cpanel = ["mediumblue", "k", "darkred", "orange"]

filedir = os.getcwd() + '/Data/'
for files in os.listdir(filedir):
    load_file = filedir+files
    data_list_ = np.loadtxt(load_file)
    # print(files, np.shape(data_list_))
    
    """
    Please ignore the redundancy. Following three sections can be easily made into a single function.
    """
    if load_file.endswith('GMM.txt') and files.startswith(dim_text):
        ### Initializing list to save composition and time variables, total count of each cluster labels, calculated probabilites and dominant label
        class_list = np.array([data_list_[:, i] for i in range(0, len(data_list_[0,:])) if i%3==1]).astype(np.float)
        class_count = []
        class_count.append(temp)
        class_count.append(comp)
        for i in range(0,4):
            class_count.append(np.transpose(np.count_nonzero(class_list == i, axis=0)))
            class_count[i+2][0]=i
        class_count_ = np.array(class_count)
        sub_class_count = class_count_[2:6, 1:].astype(float)
        max_runs = np.sum(sub_class_count[:,1])
        sub_prob = np.divide(sub_class_count, max_runs)
        sub_class_prob = [np.insert(sub_prob[i], 0, int(i), axis = 0) for i in range(0,4)]
        class_count.extend(sub_class_prob)
        main_class_count = np.amax(sub_class_count, axis = 0)
        main_class_prob = main_class_count/max_runs
        array_len = len(main_class_count)
        new_list = ['G']
        new_list_prob = ['P']
        for k in range(0,array_len):
            new_list.append(np.where(sub_class_count[:,k]==main_class_count[k])[0][0])
        new_list_prob.extend(main_class_prob)
        class_count.append(new_list); class_count.append(new_list_prob)
        #saving data
        if not os.path.isdir('Summary'):
            os.makedirs('Summary')
        np.savetxt(os.getcwd() + '/Summary/'+'summary_'+files, np.transpose(class_count), delimiter = " ", fmt='%.4s') 
        #plotting figure
        fig=plt.figure(figsize=(9,6))
        label_text = files[:-4]
        size = 100*(np.array(new_list_prob[1:])**3)
        title_text = "Color(group) and size(prob) encoded on temp-comp spread"
        plt.scatter(class_count_[0, 1:].astype(float), class_count_[1, 1:].astype(float), c=new_list[1:], cmap='coolwarm', s=size);
        custom_lines = [plt.scatter([0], [0], marker='>', color='y', label=label_text, s=100),plt.scatter([0], [0], marker='o', color='g', label='1.00', s=100), plt.scatter([0], [0], marker='o', color='g', label='0.75', s=100*.75**3), plt.scatter([0], [0], marker='o', color='g', label='0.50', s=100*.5**3)]
        plt.legend(handles=custom_lines, bbox_to_anchor=(0, 1.08), loc='upper left', ncol=4)
        plt.xlabel('Composition', fontsize=12)
        plt.ylabel('Temperature', fontsize=12)
        plt.xlim(78, 100)
        plt.ylim(20, 71)
        tru_list2 = labeling_rule(new_list[1:], len(tem_par))
        for j, val in enumerate(tru_list2):
            plt.annotate(str(int(tem_par[:,0][val]))+'%', xy=(tem_par[:,0][val], tem_par[:,1][val]), xytext=(tem_par[:,0][val], 1+tem_par[:,1][val]), ha='center', arrowprops=dict(arrowstyle="->"),)
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        fig.suptitle(title_text, fontsize=14)
        plt.savefig(os.getcwd() + '/Summary/'+'summary_'+files[:-4]+'.png', dpi=figdpi)
        plt.close()
        for i in range(0,4):
            fig=plt.figure(figsize=(9,6))
            new_text = 'Cluster'+str(i)
            label_text = files[:-4]
            size = 100*(np.array(sub_prob[i,:])**3)
            title_text = "Map of a cluster group and its probability (size)"
            plt.scatter(class_count_[0, 1:].astype(float), class_count_[1, 1:].astype(float), s=size, c=cpanel[i]);
            custom_lines = [plt.scatter([0], [0], marker='>', color='y', label=new_text, s=100),plt.scatter([0], [0], marker='o', color='g', label='1.00', s=100), plt.scatter([0], [0], marker='o', color='g', label='0.75', s=100*.75**3), plt.scatter([0], [0], marker='o', color='g', label='0.50', s=100*.5**3)]
            plt.legend(handles=custom_lines, bbox_to_anchor=(0, 1.08), loc='upper left', ncol=4)
            plt.xlabel('Composition', fontsize=12)
            plt.ylabel('Temperature', fontsize=12)
            plt.xlim(78, 100)
            plt.ylim(20, 71)
            fig.suptitle(title_text, fontsize=14)
            plt.savefig(os.getcwd() + '/Summary/Groups/'+'summary_'+files[:-4]+'_'+ new_text +'.png', dpi=figdpi)
            plt.close()

  
    if load_file.endswith('means.txt') and files.startswith(dim_text):
        ### Initializing list to save composition and time variables, total count of each cluster labels, calculated probabilites and dominant label
        class_list = np.array([data_list_[:, i] for i in range(0, len(data_list_[0,:])) if i%2==1])
        class_count = []
        class_count.append(temp)
        class_count.append(comp)
        for i in range(0,4):
            class_count.append(np.transpose(np.count_nonzero(class_list == i, axis=0)))
            class_count[i+2][0]=i
        class_count_ = np.array(class_count)
        sub_class_count = class_count_[2:6, 1:].astype(float)
        max_runs = np.sum(sub_class_count[:,1])
        sub_prob = np.divide(sub_class_count, max_runs)
        sub_class_prob = [np.insert(sub_prob[i], 0, int(i), axis = 0) for i in range(0,4)]
        class_count.extend(sub_class_prob)
        main_class_count = np.amax(sub_class_count, axis = 0)
        main_class_prob = main_class_count/max_runs
        array_len = len(main_class_count)
        new_list = ['G']
        new_list_prob = ['P']
        for k in range(0,array_len):
            new_list.append(np.where(sub_class_count[:,k]==main_class_count[k])[0][0])
        new_list_prob.extend(main_class_prob)
        class_count.append(new_list); class_count.append(new_list_prob)
        #saving data
        if not os.path.isdir('Summary'):
            os.makedirs('Summary')
        np.savetxt(os.getcwd() + '/Summary/'+'summary_'+files, np.transpose(class_count), delimiter = " ", fmt='%.4s') 
        #plotting figure
        fig=plt.figure(figsize=(9,6))
        label_text = files[:-4]
        size = 100*(np.array(new_list_prob[1:])**3)
        title_text = "Color(group) and size(prob) encoded on temp-comp spread"
        plt.scatter(class_count_[0, 1:].astype(float), class_count_[1, 1:].astype(float), c=new_list[1:], cmap='coolwarm', s=size);
        custom_lines = [plt.scatter([0], [0], marker='>', color='y', label=label_text, s=100),plt.scatter([0], [0], marker='o', color='g', label='1.00', s=100), plt.scatter([0], [0], marker='o', color='g', label='0.75', s=100*.75**3), plt.scatter([0], [0], marker='o', color='g', label='0.50', s=100*.5**3)]
        plt.legend(handles=custom_lines, bbox_to_anchor=(0, 1.08), loc='upper left', ncol=4)
        plt.xlabel('Composition', fontsize=12)
        plt.ylabel('Temperature', fontsize=12)
        plt.xlim(78, 100)
        plt.ylim(20, 71)
        tru_list2 = labeling_rule(new_list[1:], len(tem_par))
        for j, val in enumerate(tru_list2):
            plt.annotate(str(int(tem_par[:,0][val]))+'%', xy=(tem_par[:,0][val], tem_par[:,1][val]), xytext=(tem_par[:,0][val], 1+tem_par[:,1][val]), ha='center', arrowprops=dict(arrowstyle="->"),)
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        fig.suptitle(title_text, fontsize=14)
        plt.savefig(os.getcwd() + '/Summary/'+'summary_'+files[:-4]+'.png', dpi=figdpi)
        plt.close()
        for i in range(0,4):
            fig=plt.figure(figsize=(9,6))
            new_text = 'Cluster'+str(i)
            label_text = files[:-4]
            size = 100*(np.array(sub_prob[i,:])**3)
            title_text = "Map of a cluster group and its probability (size)"
            plt.scatter(class_count_[0, 1:].astype(float), class_count_[1, 1:].astype(float), s=size, c=cpanel[i]);
            custom_lines = [plt.scatter([0], [0], marker='>', color='y', label=new_text, s=100),plt.scatter([0], [0], marker='o', color='g', label='1.00', s=100), plt.scatter([0], [0], marker='o', color='g', label='0.75', s=100*.75**3), plt.scatter([0], [0], marker='o', color='g', label='0.50', s=100*.5**3)]
            plt.legend(handles=custom_lines, bbox_to_anchor=(0, 1.08), loc='upper left', ncol=4)
            plt.xlabel('Composition', fontsize=12)
            plt.ylabel('Temperature', fontsize=12)
            plt.xlim(78, 100)
            plt.ylim(20, 71)
            fig.suptitle(title_text, fontsize=14)
            plt.savefig(os.getcwd() + '/Summary/Groups/'+'summary_'+files[:-4]+'_'+ new_text +'.png', dpi=figdpi)
            plt.close()
    
    if load_file.endswith('spectral.txt') and files.startswith(dim_text):
        ### Initializing list to save composition and time variables, total count of each cluster labels, calculated probabilites and dominant label
        class_list = np.array([data_list_[:, i] for i in range(0, len(data_list_[0,:])) if i%2==1])
        class_count = []
        class_count.append(temp)
        class_count.append(comp)
        for i in range(0,4):
            class_count.append(np.transpose(np.count_nonzero(class_list == i, axis=0)))
            class_count[i+2][0]=i
        class_count_ = np.array(class_count)
        sub_class_count = class_count_[2:6, 1:].astype(float)
        max_runs = np.sum(sub_class_count[:,1])
        sub_prob = np.divide(sub_class_count, max_runs)
        sub_class_prob = [np.insert(sub_prob[i], 0, int(i), axis = 0) for i in range(0,4)]
        class_count.extend(sub_class_prob)
        main_class_count = np.amax(sub_class_count, axis = 0)
        main_class_prob = main_class_count/max_runs
        array_len = len(main_class_count)
        new_list = ['G']
        new_list_prob = ['P']
        for k in range(0,array_len):
            new_list.append(np.where(sub_class_count[:,k]==main_class_count[k])[0][0])
        new_list_prob.extend(main_class_prob)
        class_count.append(new_list); class_count.append(new_list_prob)
        #saving data
        if not os.path.isdir('Summary'):
            os.makedirs('Summary')
        np.savetxt(os.getcwd() + '/Summary/'+'summary_'+files, np.transpose(class_count), delimiter = " ", fmt='%.4s') 
        #plotting figure
        fig=plt.figure(figsize=(9,6))
        label_text = files[:-4]
        size = 100*(np.array(new_list_prob[1:])**3)
        title_text = "Color(group) and size(prob) encoded on temp-comp spread"
        plt.scatter(class_count_[0, 1:].astype(float), class_count_[1, 1:].astype(float), c=new_list[1:], cmap='coolwarm', s=size);
        custom_lines = [plt.scatter([0], [0], marker='>', color='y', label=label_text, s=100),plt.scatter([0], [0], marker='o', color='g', label='1.00', s=100), plt.scatter([0], [0], marker='o', color='g', label='0.75', s=100*.75**3), plt.scatter([0], [0], marker='o', color='g', label='0.50', s=100*.5**3)]
        plt.legend(handles=custom_lines, bbox_to_anchor=(0, 1.08), loc='upper left', ncol=4)
        plt.xlabel('Composition', fontsize=12)
        plt.ylabel('Temperature', fontsize=12)
        plt.xlim(78, 100)
        plt.ylim(20, 71)
        tru_list2 = labeling_rule(new_list[1:], len(tem_par))
        for j, val in enumerate(tru_list2):
            plt.annotate(str(int(tem_par[:,0][val]))+'%', xy=(tem_par[:,0][val], tem_par[:,1][val]), xytext=(tem_par[:,0][val], 1+tem_par[:,1][val]), ha='center', arrowprops=dict(arrowstyle="->"),)
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        fig.suptitle(title_text, fontsize=14)
        plt.savefig(os.getcwd() + '/Summary/'+'summary_'+files[:-4]+'.png', dpi=figdpi)
        plt.close()
        for i in range(0,4):
            fig=plt.figure(figsize=(9,6))
            new_text = 'Cluster'+str(i)
            label_text = files[:-4]
            size = 100*(np.array(sub_prob[i,:])**3)
            title_text = "Map of a cluster group and its probability (size)"
            plt.scatter(class_count_[0, 1:].astype(float), class_count_[1, 1:].astype(float), s=size, c=cpanel[i]);
            custom_lines = [plt.scatter([0], [0], marker='>', color='y', label=new_text, s=100),plt.scatter([0], [0], marker='o', color='g', label='1.00', s=100), plt.scatter([0], [0], marker='o', color='g', label='0.75', s=100*.75**3), plt.scatter([0], [0], marker='o', color='g', label='0.50', s=100*.5**3)]
            plt.legend(handles=custom_lines, bbox_to_anchor=(0, 1.08), loc='upper left', ncol=4)
            plt.xlabel('Composition', fontsize=12)
            plt.ylabel('Temperature', fontsize=12)
            plt.xlim(78, 100)
            plt.ylim(20, 71)
            fig.suptitle(title_text, fontsize=14)
            plt.savefig(os.getcwd() + '/Summary/Groups/'+'summary_'+files[:-4]+'_'+ new_text +'.png', dpi=figdpi)
            plt.close()
    

