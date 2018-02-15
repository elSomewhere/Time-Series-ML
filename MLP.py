#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:36:35 2018

@author: estebanlanter
"""

# load and plot dataset
from pandas import DataFrame
from pandas import read_csv
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import os
os.chdir('/Users/estebanlanter/Dropbox/Machine Learning/TimeSeries')



# load 
def parser(x):
	return datetime.strptime(x, '%Y-%m')
series = read_csv('raw_adjusted.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, sep=',')
# summarize first few rows
print(series.head())
# line plot
series.plot()
pyplot.show()

series_normalized = MinMaxScaler(feature_range=(-1,1)).fit_transform(series)
input_dat = series_normalized[:,0:6]
output_dat = series_normalized[:,[0,2,3]]
DataFrame(input_dat).plot()
DataFrame(output_dat).plot()

#we will compare LSTM vs MLP for multiple covariates for multivariate prediction. 3 independent series, and  each itself for autoregressivity will be processed
#we will leave it modeled in in low level tensorflow

#lookback window
#look forward window
#batch size: influences how many lookback windows are chosen per iteration. num iterations = epochs*samplesize/batchsize. the larger the more precise but more computation (after forward feed). If we use all samples at once, we have a good approximation of thte true distribution, but it is only one learning step. If we use smaller batch (half-size) it is 2 learning steps, but less well approximated rtue distribution. In image recognition it would be number of images per learning step.
#num_images equiv to inputsize = (samplesize-predwindow-lookback+1)
#a single batch input has following dimensions:  batchsize * lookback * numfeatures = 
#numfeatures: 3 external + 3 self = 6
#targetvector: 1d - num self * prediction window


#GDP, INFLATION, INTEREST RATES
#SELF, WEIGHTED FX, OIL, PAYROLLS, regression dummy 


#we will end with data 

#we use numpy

import numpy as np
LOOKBACK_WINDOW = 12
PREDICT_WINDOW = 4

nb_samples = len(input_dat) - LOOKBACK_WINDOW - PREDICT_WINDOW
input_list = [np.expand_dims(input_dat[i:LOOKBACK_WINDOW+i,:], axis=0) for i in range(nb_samples)] 
input_mat = np.concatenate(input_list, axis=0) #batches * lookback * numfeatuers
input_vector = input_mat.reshape(input_mat.shape[0],-1) #batches * flattened(features,lookback)
print(input_mat.shape)
print(input_vector.shape)
#prepare targets - 1d array of lengtht num_images * features. we could also justt leave this a 2d array and lett tf handle tthe rtansform
output_mat = np.asarray([output_dat[(LOOKBACK_WINDOW+i+1):(LOOKBACK_WINDOW+i+PREDICT_WINDOW+1),:] for i in range(nb_samples)]) #batches * features * horizon
output_vector = output_mat.reshape(output_mat.shape[0],-1)
print(output_mat.shape)
print(output_vector.shape)


input_mat_train = input_mat[0:50,:,:]
input_vector_train = input_vector[0:50,:]
output_mat_train = output_mat[0:50,:,:]
output_vector_train = output_vector[0:50,:]
input_mat_test = input_mat[51::,:,:]
input_vector_test = input_vector[51::,:]
output_mat_test = output_mat[51::,:,:]
output_vector_test = output_vector[51::,:]







#a function that takes input and output 3d-matrices and returns 2 generators
def BatchGenerator2(inputdat,outputdat,size,shuffle,padrest):
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    assert inputdat.shape[0]==outputdat.shape[0]
    #cut away uneven data (change this...)
    rest = inputdat.shape[0]%size
    if shuffle==True:
        shuff_input, shuff_output = unison_shuffled_copies(inputdat,outputdat)
    else:
        shuff_input = inputdat
        shuff_output = outputdat 
    if rest > 0:
        shuff_input = shuff_input[0:(shuff_input.shape[0]-rest),:,:]
        shuff_input_rest = shuff_input[(shuff_input.shape[0]-rest)::,:,:]
        shuff_output = shuff_output[0:(shuff_output.shape[0]-rest),:,:]
        shuff_output_rest = shuff_output[(shuff_output.shape[0]-rest)::,:,:]
        if padrest:
            toadd_input = np.zeros((size-rest,inputdat.shape[1],inputdat.shape[2]))
            toadd_output = np.zeros((size-rest,outputdat.shape[1],outputdat.shape[2]))
            shuff_input_rest = np.append(shuff_input_rest,toadd_input,0)
            shuff_output_rest = np.append(shuff_output_rest,toadd_output,0)     
    splitby = inputdat.shape[0]//size
    x = np.split(shuff_input,splitby)
    y = np.split(shuff_output,splitby)
    if rest > 0:
        x.append(shuff_input_rest)
        y.append(shuff_output_rest)
    out = zip(x,y)
    for i in out:
        yield i





import tensorflow as tf
from tensorflow.contrib import rnn

tf.reset_default_graph()

#for dropout
hidden_sizes = [32,64] #hidden layers
batch_size = 5
training_epochs = 10000
learning_rate=0.001
keep_prob = 1#0.9


n_input = input_mat_train.shape[1]*input_mat_train.shape[2] #number of input vars * lookbackwindow
n_output = output_mat_train.shape[1]*output_mat_train.shape[2] #number of output vars * lookbackwindow

# Store layers weight & bias
weights = [None] * (len(hidden_sizes)+1)
biases = [None] * (len(hidden_sizes)+1)
for i in range(len(hidden_sizes)):
    if i == 0:
        weights[i] = tf.Variable(tf.random_normal([n_input, hidden_sizes[i]]))
    else:
        weights[i] = tf.Variable(tf.random_normal([hidden_sizes[i-1], hidden_sizes[i]]))        
    biases[i] =     tf.Variable(tf.random_normal([hidden_sizes[i]]))
weights[-1] = tf.Variable(tf.random_normal([hidden_sizes[-1], n_output]))
biases[-1] = tf.Variable(tf.random_normal([n_output]))


num_examples = output_mat_train.shape[0]
total_batch = int(num_examples/batch_size)
lookback_window = input_mat_train.shape[1] #=time_steps
number_of_sequences = input_mat_train.shape[2]
num_output_sequences_to_predict = output_mat.shape[2]
predict_window = output_mat_train.shape[1]



###the graph
#input and output nodes
X = tf.placeholder("float", [None, lookback_window, number_of_sequences])
Y = tf.placeholder("float", [None, predict_window, num_output_sequences_to_predict])
#flatttetn to 2D - htis is a trick since tf.reshape doesn't like None dimensions
X_reshaped = tf.reshape(X,shape=(-1,tf.shape(X)[1]*tf.shape(X)[2]))
#hidden layers with nonlinear actiavtoin functions
for i in range(len(hidden_sizes)+1):
    if i == 0:
        model = tf.add(tf.matmul(X_reshaped, weights[i]), biases[i])
    else:
        model = tf.add(tf.matmul(model, weights[i]), biases[i])
    model = tf.nn.relu(model)

predictions = tf.reshape(model,[-1, predict_window, num_output_sequences_to_predict]) #output
loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(predictions, Y)))
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
#accuracy - this tests against train set
accuracy = tf.sqrt(tf.reduce_mean(tf.squared_difference(predictions, Y)))
keep_prob_node = tf.placeholder(tf.float32, name='keep_prob')


import matplotlib.pyplot as plt
plotter_acc_train = [np.nan] * training_epochs
plotter_acc_val = [np.nan] * training_epochs
xplot = np.arange(0,training_epochs)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_batches = num_examples//batch_size
    for epoch in range(training_epochs):
        print(epoch)
        train_acc = []
        for i, (x, y) in enumerate(BatchGenerator2(input_mat_train,output_mat_train,batch_size,True,True), 1):
            # Run optimization op (backprop) and cost op (to get loss value)
            #we output train node, loss node and actual final model layer
            loss_, _,  batch_acc = sess.run([loss, optimizer, accuracy], feed_dict={X: x,Y: y})
            train_acc.append(batch_acc)
            plotter_acc_train[epoch] = batch_acc
            if (i + 1) % num_batches == 0: #if we are in the final step
                val_acc = []
                for xx, yy in BatchGenerator2(input_mat_test,output_mat_test,batch_size,True,True):
                    val_batch_acc = sess.run(accuracy, feed_dict={X: xx,Y: yy})
                    val_acc.append(val_batch_acc)
                    plotter_acc_val[epoch] = val_batch_acc
                    print(sess.run(predictions, feed_dict={X: xx,Y: yy}))
                    print(yy)
                print("Epoch: {}/{}...".format(epoch+1, training_epochs),
                      "Batch: {}/{}...".format(i+1, num_batches),
                      "Train Loss: {:.3f}...".format(loss_),
                      "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                      "Val Accuracy: {:.3f}".format(np.mean(val_acc)))
        plt.plot(xplot,np.array(plotter_acc_train),'r')
        plt.plot(xplot,np.array(plotter_acc_val),'b')
        plt.xlim(0, training_epochs)
        plt.ylim(0,100)
        plt.show()
        #plt.pause(0.05)
    saver.save(sess, "checkpoints/MLP_state.ckpt")


