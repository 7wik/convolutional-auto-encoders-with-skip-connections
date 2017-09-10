# parag k mittal's code on convolutional auto encoder was very helpful in writing this code. 
# His tutorials has laid the basis for my understanding of basics in tensorflow

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import*
import os
batch_size = 100
n_epochs = 10
learning_rate = 0.01
n_examples = 10
filters=[1, 8, 16, 32]#no. of filters in each layer(first element= number of channels in input image)
filter_sizes=[3, 3, 3]
def lrelu(x, leak=0.2, name="lrelu"):#function for  relu
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
	        f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def corrupt(x):#function for corrupting the input
	return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),minval=0,maxval=2,dtype=tf.int32), tf.float32))

def load_mnist():
	data_dir = os.path.join("./data-1", "mnist")
	# data_dir="/home/satwik/Desktop/swaayatt_satwik/gan_test_Code/data /mnist"

	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000)).astype(np.float)

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000)).astype(np.float)

	trY = np.asarray(trY)
	teY = np.asarray(teY)

	X = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0).astype(np.int)

	seed = 547
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)

	y_vec = np.zeros((len(y), 10), dtype=np.float)
	for i, label in enumerate(y):
		y_vec[i,y[i]] = 1.0

	return (X/255.),y_vec

def autoencoder(input_shape=[batch_size, 28,28,1],n_filters=filters,filter_sizes=filter_sizes,corruption=False):
	x = tf.placeholder(tf.float32, input_shape, name='x')
	# x_tensor=tf.reshape(x,[100,32,32,n_filters[0]])#reshaped input
	current_input=x

	#corupting the image
	if corruption:
		p = corrupt(current_input)
		current_input=p
	
	encoder = []#list for holding weights
	
	shapes = []#list for holding shapes of output layers
	
	outputs = []
	
	#encoding

	for layer_i, n_output in enumerate(n_filters[1:]):
		
		n_input = current_input.get_shape().as_list()[3]
		
		shapes.append(current_input.get_shape().as_list())
		
		W = tf.Variable(tf.random_uniform([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output],-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))#creating a variable
		
		b = tf.Variable(tf.zeros([n_output]))#creating variable
		
		encoder.append(W)
		
		output = lrelu(tf.add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
		
		outputs.append(output)
		
		current_input = output
	#skip connections and decoding

	B = tf.Variable(tf.zeros([n_output]))
	
	n_input = current_input.get_shape().as_list()[3]
	
	b_= tf.Variable(tf.zeros([n_output]))
	
	t=current_input.get_shape().as_list()
	
	w= tf.Variable(tf.random_uniform((t),-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))
	
	outputs[-1] = tf.multiply(tf.add(outputs[-1],B),lrelu(tf.add(tf.multiply(current_input,w),b_)))
	
	current_input=outputs[-1]
	
	z = current_input
	
	encoder.reverse()

	shapes.reverse()
	
	outputs.reverse()
	
	for layer_i, shape in enumerate(shapes):
		
		W = encoder[layer_i]
		
		n_input = current_input.get_shape().as_list()[3]
		
		b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
		
		if (layer_i < len(filter_sizes)-1):
		
			B = tf.Variable(tf.zeros(filters[len(filter_sizes)-layer_i-1]))
			
			b_= tf.Variable(tf.zeros(filters[len(filter_sizes)-layer_i-1]))
			
			w= tf.Variable(tf.random_uniform(shape=outputs[layer_i+1].get_shape().as_list(),minval=-1.0 / math.sqrt(n_input),maxval=1.0 / math.sqrt(n_input)))
			
			k = tf.multiply(tf.add(outputs[layer_i+1],B),lrelu(tf.add(tf.multiply(outputs[layer_i+1],w),b_)))
			
			output=lrelu(tf.add(tf.nn.conv2d_transpose(outputs[layer_i],W,tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),strides=[1, 2, 2, 1], padding='SAME'), b))
			
			output=tf.add(k,output)
		else:
			
			output=lrelu(tf.add(tf.nn.conv2d_transpose(outputs[layer_i],W,tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),strides=[1, 2, 2, 1], padding='SAME'), b))
		
		current_input = output
	
	y = current_input
	
	cost = tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(y - x,[batch_size,-1])),axis=1))#MSE
	
	return {'x': x, 'z': z, 'y': y, 'cost': cost,'p':p}

trax,tray=load_mnist()

tex=trax[:10000]

tey=tray[:10000]

trax=trax[10000:]

tray=tray[10000:]

ae = autoencoder(corruption=True)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for epoch_i in range(n_epochs):
	
	for batch_i in range(trax.shape[0] // batch_size):
		
		batch_xs = trax[batch_i*(batch_size):(batch_i+1)*(batch_size)]
		
		train = np.asarray((batch_xs))
		
		sess.run(optimizer, feed_dict={ae['x']: train})
	
	print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

test_xs= tex[:100]

test_xs_norm = np.array(test_xs)

recon,corrupted = sess.run([ae['y'],ae['p']], feed_dict={ae['x']: test_xs_norm})

save_images(test_xs_norm, image_manifold_size(test_xs_norm.shape[0]),'./{}/original.png'.format("skip_connections"))
save_images(recon, image_manifold_size(recon.shape[0]),'./{}/reconstructed.png'.format("skip_connections"))
save_images(corrupted, image_manifold_size(corrupted.shape[0]),'./{}/corrupted.png'.format("skip_connections"))
print(recon.shape)

	
	
