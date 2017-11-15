

import scipy.misc
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

prefix = './Datas/'
def get_img(img_path, crop_h, resize_h):
	img=scipy.misc.imread(img_path).astype(np.float)
	# crop resize
	crop_w = crop_h
	#resize_h = 64
	resize_w = resize_h
	h, w = img.shape[:2]
	j = int(round((h - crop_h)/2.))
	i = int(round((w - crop_w)/2.))
	cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])

	return np.array(cropped_image)/255.0


class mnist():
	def __init__(self, flag='conv', is_tanh = False):
		datapath = prefix + 'mnist'
		self.X_dim = 784
		self.z_dim = 100
		self.y_dim = 10
		self.size = 28
		self.channel = 1
		self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):
		batch_imgs,y = self.data.train.next_batch(batch_size)

		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel))
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1
		return batch_imgs, y

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		# fig = plt.figure(figsize=(4, 4))
		# gs = gridspec.GridSpec(4, 4)
		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(8, 8)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig



