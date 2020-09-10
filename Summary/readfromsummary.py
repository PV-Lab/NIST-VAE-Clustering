# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:55:48 2020

@author: Janak MIT
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

comp = 85
temp = 68
sumfile ="D:/MIT/XRD/NIST-Jae/Summary/summary_2d_threek_means.txt"
src_wb = np.transpose(np.loadtxt(sumfile, delimiter=" ", skiprows=1))
# print(src_wb[1:,1])
fig = plt.figure(figsize=(5,5))
index_list = np.where(src_wb[1,:]==temp)
clist = ['darkcyan', 'olive', 'black']
for i in range(6,9):
    plt.scatter(src_wb[0,index_list], src_wb[i,index_list], label='class '+str(i-6), c=clist[i-6], s=20) 
plt.hlines(y=.5, xmin=78, xmax=100, linestyle='dashed', color='red', alpha=0.5)
plt.xlabel('Composition', fontsize = 15)
plt.ylabel('Probability', fontsize=15)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.xlim(78.5, 100)
# fig.suptitle(title, fontsize = 12.5)
plt.legend(loc='upper right')
plt.savefig(os.getcwd() +"/"+ str(temp)+"percent" +"map"+'.png', dpi=600)
plt.show()   


