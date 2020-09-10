# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:42:34 2020

@author: Janak MIT
"""

# -*- coding: utf-8 -*-
"""
Parts of this file were imported from another python file created by Danny in PV-lab

"""
"""
This file extracts data from the provided combiview text files in this current folder, performs VAE and clustering on the extracted data and saves the resulting clustering labels. 
Use class_averaging file for reading the outputs from this file and plotting them

"""
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import time


### limiting the range of data to the region where XRD peak exists
range_min = 1800;
range_max = 2150;

### defining function to randomize the order of XRD samples/compositions, also allows selecting how many samples to draw
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


### Selecting which compositions to use, max 44 selects all the way from 99 to min 56, works without error from 1-43, need to check 44
sel_range = []
sel_k = 0
ul_ = 20
for i in range(0,8):
    for j in range(0,ul_):
        sel_range.append(sel_k)
        sel_k += 1
    sel_k += 44-ul_

### Reading from the provided XRD text file
theta = np.transpose(np.loadtxt('VO2 -Nb2O3 XRD Combiview.txt', max_rows=1))[range_min:range_max]

raw_data = np.loadtxt('VO2 -Nb2O3 XRD Combiview.txt', skiprows=1)[sel_range,range_min:range_max]

### Checking features of data for early overview as well as possible ways of normalization
maxraw = [max(i) for i in raw_data]
minraw = [min(i) for i in raw_data]

zero_based = [] #initializing a list that will be placeholder for baseline subtracted data
newlist = []

### Doing baseline subtraction
for i in range(len(raw_data)):
    array_ = raw_data[i]-minraw[i]
    zero_based.append(array_)
    
# max_fac = max([max(i) for i in zero_based])
# std_ = [np.std(i) for i in zero_based]
# avg_fac = [np.average(i) for i in zero_based]
# xyz = zero_based
# zero_based = np.transpose(zero_based)

for i in range(len(zero_based)):
    ### Three different ways normalize and/or scale, first and second are similar. Third method is just scaling using maximum XRD value. 
    # newarray = normalize((zero_based[i])[:,np.newaxis], axis=0).ravel() 
    newarray = zero_based[i]/np.linalg.norm(zero_based[i])
    # newarray = zero_based[i]/max_fac  
    newlist.append(newarray)
# mat_shape = np.shape(np.transpose(newlist))
# templist = (np.array(np.transpose(newlist))).flatten()
# normed = templist/np.linalg.norm(templist)
# zero_based = xyz
# a=np.transpose(np.array(newlist))

a = np.array(newlist) #Normalized data converted to numpy array

### Reading from the provided composition and temperature index file
tem_par = np.loadtxt('VO2 - Nb2O3 Composition and temp Combiview.txt', skiprows=1)[sel_range]

all_par = [np.insert(tem_par[i], 0, i+1) for i in range(0,len(tem_par))] #introducing a new list that enumerates temp/composition combination
d_par = np.concatenate(all_par, axis=0)
par = d_par.reshape(len(d_par)//len(all_par[0]), len(all_par[0]))


b = a #creating a copy of numpy array for reference later on
Dim = len(a[0]) #inputting full array of samples


### further variables before using tensorflow
tf.reset_default_graph()

BATCH_SIZE = 140 #determines a number to extract out of total available
LR = 0.0005  

X_in = tf.placeholder(dtype=tf.float32, shape=[None, Dim], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, Dim], name='Y')


dec_in_channels = 1
n_latent = 2 #latent space dimensions


### Defining encoder/decoder functions
def encoder(X_in):
   
    with tf.variable_scope("encoder", reuse=None):

        en0 = tf.layers.dense(X_in, 128, tf.nn.relu) #less than Dim
        en1 = tf.layers.dense(en0, 64, tf.nn.relu)
        x = tf.layers.dense(en1, 32, tf.nn.relu)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 *( tf.layers.dense(x, units=n_latent))# positive
        epsilon =  tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z):
    with tf.variable_scope("decoder", reuse=None):
        ### inverse order of encoding
        de0 = tf.layers.dense(encoded, 32, tf.nn.relu)
        de1 = tf.layers.dense(de0, 64, tf.nn.relu)
        de2 = tf.layers.dense(de1, 128, tf.nn.relu)
        decoded = tf.layers.dense(de2, Dim, tf.nn.softplus)

        return decoded


encoded, mn, sd = encoder(X_in)
decoded = decoder(encoded)


### Initializing some output lists
wcss_list = []
aic_list = []
bic_list = []

indx_low = 3 # cutoff range below which figures won't be labelled; its implementations have been commented out

### Ensuring folder to save output exists
if not os.path.exists('Data'):
    os.makedirs('Data')

# Running encoding/decoding, clustering and outputting results for 150 times
for run_i in range(150):
    ### Initializing lists for outputting data after each run
    twok_means = []
    fourGMM = []
    threek_means = []
    fourk_means = []
    twoGMM = []
    threeGMM = []
    
    two_spectral = []
    three_spectral = []
    four_spectral = []
    twod_spectral = []
    threed_spectral = []
    fourd_spectral = []
    

    img_loss = 1*tf.reduce_sum(tf.squared_difference(decoded, X_in), 1) #You can play with your the factor (1 right now) for better loss optimization
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)

    loss = tf.reduce_mean(img_loss + latent_loss)
    train = tf.train.AdamOptimizer(LR).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    conv_x = []
    conv_y = []
    
    ### Loss optimization over 22000 loops
    for step in range(22000):
        b_x, b_y = next_batch(BATCH_SIZE, b, par)
        _, encoded_, mn_,sd_,decoded_, loss_ = sess.run([train, encoded,mn,sd, decoded, loss], {X_in: b_x})
    
        if step % 100 == 0:     # plotting
           conv_x.append(step)
           conv_y.append(loss_)
           print('train loss: %.4f' % loss_)
           

    view_data = b
    encoded_data = sess.run(encoded, {X_in: view_data})
    decoded_data = sess.run(decoded, {X_in: view_data})
    
    
    ### Plotting figures - loss optimization
    fig = plt.figure(figsize=(14,10))
    plt.scatter(conv_x[1:],conv_y[1:], label='Steps vs Loss')
    plt.ylabel('Loss',fontsize='xx-large')
    plt.xlabel('Steps',fontsize='xx-large')
    plt.savefig(os.getcwd() + '/plots/' + 'tempcomp-Steps-vs-loss' +'.png')
    plt.close()
    
    
    print('Done with VAE part, now doing clustering on run ' + str(run_i+1)) #just a line to show number of VAE runs done. 
    
    
    ####Clustering Below####
    ##Code is clunky and repititious need to clean with some functions :(
    Xtemp = []
    for i in range(0,len(encoded_data[1,:])):
        Xtemp.append(normalize(encoded_data[:,i][:,np.newaxis], axis=0).ravel()) #normalizing VAE output
    X = np.transpose(np.array(Xtemp))
    
    cluster_data_ = len(par)
    
    #K-means clustering
    def kmeans_plot(x, y, ncl=3, title = "K-means Clustering"):
        kmeans1 = KMeans(n_clusters=ncl, init='k-means++', max_iter=300, n_init=10, random_state=0)
        pred_y = kmeans1.fit_predict(y)
        
        #Use the section below if you want to plot/save images from every run
        
        '''
        indx_low = 3
        indx_high = ul_-1
        tru_list = []
        colbox = kmeans1.labels_
        while indx_high < cluster_data_:
            for indx in range(indx_low, indx_high):
                if colbox[indx] != colbox[indx+1]:
                    if colbox[indx] == colbox[indx-1] and colbox[indx]==colbox[indx-2]: 
                        tru_list.append(indx+1)
            indx_low += ul_
            indx_high += ul_
        fig = plt.figure(figsize=(20,8))
        plt.subplot(121)
        plt.scatter(y[:,0], y[:,1], c=kmeans1.labels_, cmap='coolwarm', label='k-means, '+str(ncl)+' clusters')
        plt.legend(bbox_to_anchor=(0, 1.06), loc='upper left', ncol=1)
        plt.xlabel("VAE1", fontsize=10)
        plt.ylabel("VAE2", fontsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.subplot(122)
        plt.scatter(x[:,0], x[:,1], c=kmeans1.labels_, cmap='coolwarm', label='k-means, '+str(ncl)+' clusters')
        plt.legend(bbox_to_anchor=(0, 1.06), loc='upper left', ncol=1)
        plt.ylabel('Temperature('+u"\u2103"+")", fontsize = 10)
        plt.xlabel('Composition(%)', fontsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        for i, val in enumerate(tru_list):
            plt.annotate(str(int(x[:,0][val]))+'%', xy=(x[:,0][val], x[:,1][val]), xytext=(x[:,0][val], 1+x[:,1][val]), ha='center', arrowprops=dict(arrowstyle="->"),)
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        fig.suptitle(title, fontsize = 16)
        plt.savefig(os.getcwd() + '/plots/' + str(n_latent)+"d_"+title[:8]+"_"+str(ncl) +'.png')
        plt.close()
        '''
        ### Assigning clustering labels and ordered clustering labels to variables for output
        label_data = [float(run_i)]
        relabel_data = [float(run_i)]
        relabelled_list = []
        uni_vals = []
        for k in kmeans1.labels_:
            if k not in uni_vals:
                uni_vals.append(k)
        for k in kmeans1.labels_:
            relabelled_list.append(np.where(uni_vals==k)[0][0])
        label_data.extend(kmeans1.labels_)
        relabel_data.extend(relabelled_list)
        if ncl == 2:
            twok_means.append(label_data)
            twok_means.append(relabel_data)
        if ncl == 3:
            threek_means.append(label_data)
            threek_means.append(relabel_data)
        if ncl == 4:
            fourk_means.append(label_data)
            fourk_means.append(relabel_data)
            
    
    # Assigning values for elbow plot    
    wcss = []
    for i in range(1, 20):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    wcss_list.append(wcss)
    # fig = plt.figure(figsize=(12,10))
    # plt.plot(range(1, 15), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.rc('xtick', labelsize=10)
    # plt.rc('ytick', labelsize=10)
    # plt.savefig(os.getcwd() + '/plots/' + str(n_latent)+"d_"+ 'k-means_WCSS_clusters' +'.png')
    # plt.close()
    
    # Running and plotting k-means clustering with different cluster numbers
    kmeans_plot(par[:,1:], X, 2)
    kmeans_plot(par[:,1:], X, 3)
    kmeans_plot(par[:,1:], X, 4)
    
        
    #GMM cluster - similar method as k-means for clustering, assigning labels, ordering labels and saving to lists.
    n_components = np.arange(1, 11, 1)
    cov_type = 'diag'
    models = [GMM(n, covariance_type=cov_type, random_state=0).fit(X)
              for n in n_components]
    
    aic_list.append([m.aic(X) for m in models])
    bic_list.append([m.bic(X) for m in models])
    # fig = plt.figure(161, figsize=(12,10))
    # plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    # plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    # plt.legend(loc='best')
    # plt.xlabel('Number of Components');
    # plt.rc('xtick', labelsize=10)
    # plt.rc('ytick', labelsize=10)
    # fig.suptitle('GMM BIC/AIC values', fontsize=16)
    # plt.savefig(os.getcwd() + '/plots/' + str(n_latent)+"d_"+'gmm_bic-aic_'+cov_type+'.png')
    # plt.close()
    
    
    for i in range(2,5):
        gmm = GMM(n_components=i, covariance_type=cov_type, random_state=0).fit(X)
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        indx_high = ul_-1
        tru_list2 = []
        colbox3 = labels
        
        '''
        while indx_high < cluster_data_:
            for indx in range(indx_low, indx_high):
                if colbox3[indx] != colbox3[indx+1]:
                    if colbox3[indx] == colbox3[indx-1] and colbox3[indx]==colbox3[indx-2]: 
                        tru_list2.append(indx+1)
            indx_low += ul_
            indx_high += ul_
        size = 40 * probs.max(1) ** 2  # square emphasizes differences
        fig = plt.figure(162+i, figsize=(20,8))
        plt.subplot(121)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=size, label='N_comp='+str(i));
        plt.legend(bbox_to_anchor=(0, 1.06), loc='upper left', ncol=1)
        plt.xlabel('VAE1', fontsize=10)
        plt.ylabel('VAE2', fontsize=10)
        plt.subplot(122)
        plt.scatter(par[:,1], par[:,2], c=labels, cmap='coolwarm', s=size, label='GMM, n_comp = '+str(i))
        plt.legend(bbox_to_anchor=(0, 1.06), loc='upper left', ncol=1)
        plt.ylabel('Temperature('+u"\u2103"+")", fontsize = 10)
        plt.xlabel('Composition(%)', fontsize=10)
        for j, val in enumerate(tru_list2):
            plt.annotate(str(int(par[:,1][val]))+'%', xy=(par[:,1][val], par[:,2][val]), xytext=(par[:,1][val], 1+par[:,2][val]), ha='center', arrowprops=dict(arrowstyle="->"),)
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        fig.suptitle("GMM clustering on latent space")
        plt.savefig(os.getcwd() + '/plots/' + str(n_latent)+"d_"+ 'gmm_clusters-'+cov_type+str(i)+'.png')
        plt.close()
        ''' 
        
        newlabels_ = [float(run_i)]
        newlabels_.extend(labels)
        relabel_data_ = [float(run_i)]
        relabelled_list_ = []
        uni_vals_ = []
        for k in labels:
            if k not in uni_vals_:
                uni_vals_.append(k)
        for k in labels:
            relabelled_list_.append(np.where(uni_vals_==k)[0][0])
        relabel_data_.extend(relabelled_list_)
        labelprobs_ = [float(run_i)]
        labelprobs_.extend(probs.max(1))
        if i == 2:
            twoGMM.append(newlabels_)
            twoGMM.append(relabel_data_)
            twoGMM.append(labelprobs_)
        if i == 3:
            threeGMM.append(newlabels_)
            threeGMM.append(relabel_data_)
            threeGMM.append(labelprobs_)
        if i == 4:
            fourGMM.append(newlabels_)
            fourGMM.append(relabel_data_)
            fourGMM.append(labelprobs_)
            
            
    #Spectral clustering - similar method as k-means and GMM
    def spectral_plot(x, y, ncl=3, alabels="kmeans", title = "Spectral Clustering"):
        scluster = SpectralClustering(n_clusters=ncl, affinity='rbf', assign_labels=alabels, random_state=0)
        pred_y = scluster.fit_predict(y)
        
        '''
        indx_low = 3
        indx_high = ul_-1
        tru_list = []
        colbox = scluster.labels_
        while indx_high < cluster_data_:
            for indx in range(indx_low, indx_high):
                if colbox[indx] != colbox[indx+1]:
                    if colbox[indx] == colbox[indx-1] and colbox[indx]==colbox[indx-2]: 
                        tru_list.append(indx+1)
            indx_low += ul_
            indx_high += ul_
        fig = plt.figure(figsize=(20,8))
        plt.subplot(121)
        plt.scatter(y[:,0], y[:,1], c=scluster.labels_, cmap='coolwarm', label='Spectral, '+ str(alabels)+" "+str(ncl)+' clusters')
        plt.legend(bbox_to_anchor=(0, 1.06), loc='upper left', ncol=1)
        plt.xlabel("VAE1", fontsize=10)
        plt.ylabel("VAE2", fontsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.subplot(122)
        plt.scatter(x[:,0], x[:,1], c=scluster.labels_, cmap='coolwarm', label='Spectral, '+ str(alabels)+" "+str(ncl)+' clusters')
        plt.legend(bbox_to_anchor=(0, 1.06), loc='upper left', ncol=1)
        plt.ylabel('Temperature('+u"\u2103"+")", fontsize = 10)
        plt.xlabel('Composition(%)', fontsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        for i, val in enumerate(tru_list):
            plt.annotate(str(int(x[:,0][val]))+'%', xy=(x[:,0][val], x[:,1][val]), xytext=(x[:,0][val], 1+x[:,1][val]), ha='center', arrowprops=dict(arrowstyle="->"),)
        fig.suptitle(title, fontsize = 16)
        plt.savefig(os.getcwd() + '/plots/' + str(n_latent)+"d_"+title[:8]+"_"+str(ncl)+"_"+str(alabels) +'.png')
        plt.close()
        '''
        
        label_data = [float(run_i)]
        relabel_data = [float(run_i)]
        relabelled_list = []
        uni_vals = []
        for k in scluster.labels_:
            if k not in uni_vals:
                uni_vals.append(k)
        for k in scluster.labels_:
            relabelled_list.append(np.where(uni_vals==k)[0][0])
        label_data.extend(scluster.labels_)
        relabel_data.extend(relabelled_list)
        if ncl == 2:
            if alabels =="kmeans":
                two_spectral.append(label_data)
                two_spectral.append(relabel_data)
            if alabels == "discretize":
                twod_spectral.append(label_data)
                twod_spectral.append(relabel_data)
        if ncl == 3:
            if alabels =="kmeans":            
                three_spectral.append(label_data)
                three_spectral.append(relabel_data)
            if alabels == "discretize":
                threed_spectral.append(label_data)
                threed_spectral.append(relabel_data)
        if ncl == 4:
            if alabels =="kmeans":            
                four_spectral.append(label_data)
                four_spectral.append(relabel_data)
            if alabels == "discretize":
                fourd_spectral.append(label_data)
                fourd_spectral.append(relabel_data)
        
    
    #running and plotting spectral clustering with k-means and discretize method
    spectral_plot(par[:,1:], X, 2, "kmeans")
    spectral_plot(par[:,1:], X, 3, "kmeans")
    spectral_plot(par[:,1:], X, 4, "kmeans")
    spectral_plot(par[:,1:], X, 2, "discretize")
    spectral_plot(par[:,1:], X, 3, "discretize")
    spectral_plot(par[:,1:], X, 4, "discretize")
    
    ### Just writing and updating the save files after each run
    
    if run_i>0:
        twok_means_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'twok_means.txt', delimiter = " ")
        twok_means = np.append(np.transpose(twok_means_), twok_means, axis=0)
        threek_means_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'threek_means.txt', delimiter = " ")
        threek_means = np.append(np.transpose(threek_means_), threek_means, axis=0)
        fourk_means_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'fourk_means.txt', delimiter = " ")
        fourk_means = np.append(np.transpose(fourk_means_), fourk_means, axis=0)
        twoGMM_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'twoGMM.txt', delimiter = " ")
        twoGMM = np.append(np.transpose(twoGMM_), twoGMM, axis=0)
        threeGMM_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'threeGMM.txt', delimiter = " ")
        threeGMM = np.append(np.transpose(threeGMM_), threeGMM, axis=0)
        fourGMM_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'fourGMM.txt', delimiter = " ")
        fourGMM = np.append(np.transpose(fourGMM_), fourGMM, axis=0)
        
        two_spectral_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'two_spectral.txt', delimiter = " ")
        two_spectral = np.append(np.transpose(two_spectral_), two_spectral, axis=0)
        three_spectral_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'three_spectral.txt', delimiter = " ")
        three_spectral = np.append(np.transpose(three_spectral_), three_spectral, axis=0)
        four_spectral_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'four_spectral.txt', delimiter = " ")
        four_spectral = np.append(np.transpose(four_spectral_), four_spectral, axis=0)
        twod_spectral_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'twod_spectral.txt', delimiter = " ")
        twod_spectral = np.append(np.transpose(twod_spectral_), twod_spectral, axis=0)
        threed_spectral_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'threed_spectral.txt', delimiter = " ")
        threed_spectral = np.append(np.transpose(threed_spectral_), threed_spectral, axis=0)
        fourd_spectral_ = np.loadtxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'fourd_spectral.txt', delimiter = " ")
        fourd_spectral = np.append(np.transpose(fourd_spectral_), fourd_spectral, axis=0)
         
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'twok_means.txt', np.transpose(twok_means), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'threek_means.txt', np.transpose(threek_means), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'fourk_means.txt', np.transpose(fourk_means), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'twoGMM.txt',np.transpose(twoGMM), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'threeGMM.txt',np.transpose(threeGMM), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'fourGMM.txt',np.transpose(fourGMM), delimiter = " ", fmt='%.4s')

    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'two_spectral.txt', np.transpose(two_spectral), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'three_spectral.txt', np.transpose(three_spectral), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'four_spectral.txt', np.transpose(four_spectral), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'twod_spectral.txt', np.transpose(twod_spectral), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'threed_spectral.txt', np.transpose(threed_spectral), delimiter = " ", fmt='%.4s')
    np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'fourd_spectral.txt', np.transpose(fourd_spectral), delimiter = " ", fmt='%.4s')
 
    sl_time = 15 #Letting computer take a break after some runs
    if run_i < 149:
        print('Sleeping for '+ str(sl_time) +' secs, done with run '+str(run_i)) 
        time.sleep(sl_time)
    if run_i%25 == 13:
        time.sleep(sl_time*4)

np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'wcss.txt',np.transpose(wcss_list), delimiter = " ", fmt='%.4s')
np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'aic.txt',np.transpose(aic_list), delimiter = " ", fmt='%.4s')
np.savetxt(os.getcwd() + '/Data/'+ str(n_latent)+"d_"+'bic.txt',np.transpose(bic_list), delimiter = " ", fmt='%.4s')    
    
