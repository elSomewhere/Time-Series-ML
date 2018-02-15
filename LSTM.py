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
output_dat = series_normalized[:,[0]]
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
        shuff_input = shuff_input[0:(shuff_input.shape[0]-rest)]
        shuff_input_rest = shuff_input[(shuff_input.shape[0]-rest)::]
        shuff_output = shuff_output[0:(shuff_output.shape[0]-rest)]
        shuff_output_rest = shuff_output[(shuff_output.shape[0]-rest)::]
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


#for dropout
lstm_sizes = [512,512] #hidden layers
batch_size = 1
training_epochs = 10000
learning_rate=0.001
keep_prob = 1#0.9

num_examples = output_mat_train.shape[0]
total_batch = int(num_examples/batch_size)
lookback_window = input_mat_train.shape[1] #=time_steps
number_of_sequences = input_mat_train.shape[2]
num_output_sequences_to_predict = output_mat.shape[2]
predict_window = output_mat_train.shape[1]

tf.reset_default_graph()

X = tf.placeholder("float", [batch_size, lookback_window, number_of_sequences]) # or X = tf.placeholder("float", [batch_size, lookback_window*number_of_sequences]) - IS IT SO????
Y = tf.placeholder("float", [batch_size, predict_window, num_output_sequences_to_predict])

#resize X to number_of_sequences*t[batch_size,lookback_window] or t[batch_size,lookback_window,number_of_sequences]

keep_prob_node = tf.placeholder(tf.float32, name='keep_prob')
# Forward passes# Forward passes
lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]
drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_node) for lstm in lstms] #dropout layer
cell = tf.contrib.rnn.MultiRNNCell(drops) #stacks layers together - is still a RNNCell, inherits rom it
initial_state = cell.zero_state(batch_size, tf.float32)
lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state) #Creates a recurrent neural network specified by RNNCell cell. Everytime called, it returns model output as wellas final statets
#Training / prediction / loss
out_weights=tf.Variable(tf.random_normal([lstm_sizes[-1],num_output_sequences_to_predict*predict_window]))
out_bias=tf.Variable(tf.random_normal([num_output_sequences_to_predict*predict_window]))
#we also reshape back here to the 3d form of inputs/outputs
predictions=tf.reshape(tf.matmul(lstm_outputs[:, -1],out_weights)+out_bias,[batch_size, predict_window, num_output_sequences_to_predict]) #we only care about the last layer of te unrolled network. Also, this here we have basically created a manual final hidden layer +bias term. No activation function needed (equal tto linear activation). We cuold have used this also: tf.contrib.layers.fully_connected(lstm_outputs[:, -1], 1, activation_fn=None)
#lstm_outputs[:, -1] is final unrolled layer which we will input into a linear activation function
loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(predictions, Y)))
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
#accuracy - this tests against train set
accuracy = tf.sqrt(tf.reduce_mean(tf.squared_difference(predictions, Y)))


#optimizers - out_weights and out_bias enters here. they are part of predictions, which are part o the loss op. All nodes making up loss_op are partt of the bpp optimization. therefore, thtey will be changed




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
        state = sess.run(initial_state)
        train_acc = []
        for i, (x, y) in enumerate(BatchGenerator2(input_mat_train,output_mat_train,batch_size,False,True), 1):
            feed = {X: x,Y: y,keep_prob_node: keep_prob,initial_state: state}
            loss_, state, _,  batch_acc = sess.run([loss, final_state, optimizer, accuracy], feed_dict=feed)
            train_acc.append(batch_acc)
            plotter_acc_train[epoch] = batch_acc
            if (i + 1) % num_batches == 0: #if we are in the final step
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for xx, yy in BatchGenerator2(input_mat_test,output_mat_test,batch_size,False,True):
                    feed = {X: xx,Y: yy,keep_prob_node: 1,initial_state: val_state}
                    val_batch_acc, val_state, pred = sess.run([accuracy, final_state,predictions], feed_dict=feed)
                    val_acc.append(val_batch_acc)
                    plotter_acc_val[epoch] = val_batch_acc
                    print(pred)
                    print('____________')
                    print(yy)
                print("Epoch: {}/{}...".format(epoch+1, training_epochs),
                      "Batch: {}/{}...".format(i+1, num_batches),
                      "Train Loss: {:.3f}...".format(loss_),
                      "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                      "Val Accuracy: {:.3f}".format(np.mean(val_acc)))
        plt.plot(xplot,np.array(plotter_acc_train),'r')
        plt.plot(xplot,np.array(plotter_acc_val),'b')
        plt.xlim(0, training_epochs)
        plt.ylim(0,2)
        plt.show()
        #plt.pause(0.05)
    saver.save(sess, "checkpoints/lstm_state.ckpt")





#stateful: hidden memory state remains between successive batches. Only once al batches are used & epoch is over is it reset. Default here
#stateless: the state within the network is reset after each training batch of batch_size samples
#stateful: the state within the network is maintained after each training batch of batch_size samples - deafult here
#If we want to implement larger batches - eithter reset within loop of batches (inner loop). Or live hte fact that there is optimizattion of distance = batch size between time points


#inputs for a gen per news event:
#target data history
#body of other econ series
#calendar evevnts window (past / future)

#outputs
#binned text body