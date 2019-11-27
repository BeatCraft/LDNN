#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from sklearn import svm, metrics
from sklearn.externals import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
#import mglearn

# 学習用データの数
SIZE_TRAINING = 5000

# 検証用データの数
SIZE_TEST = 500



def load_data(type_, size):
	"""
		type_ : training / test
		size : 返却する要素数
	"""
	#csvデータを改行してデータ数分区切っていく
	with open(os.path.join("csv", "%s_image.csv" % type_)) as f:
		images = f.read().split("\n")[:size]
	with open(os.path.join("csv", "%s_label.csv" % type_)) as f:
		labels = f.read().split("\n")[:size]

	#各ピクセルに相当する数値（白黒：白0～黒255）を256で割って0-1の値に変換
	images = [[int(i)/256 for i in image.split(",")] for image in images]
	labels = [int(l) for l in labels]

	return images, labels

colors = [
		"#000000",#000
		"#FF0000",
		"#00FF00",
		"#0000FF",
		"#FF00FF",
		"#FFFF00",
		"#000000",
		"#0F0F0F",
		"#A00000",
		"#00A000",
		"#0000A0",
		]
def draw(title, data, labels, name, xlim=None, ylim=None):
	plt.figure(figsize=(10,10))
	plt.title(title)

	if xlim is None:
		plt.xlim(data[:,0].min(), data[:,0].max())
	else:
		plt.xlim(xlim[0], xlim[1])

	if ylim is None:
		plt.ylim(data[:,1].min(), data[:,1].max())
	else:
		plt.ylim(ylim[0], ylim[1])


	for i in range(len(data)):
		plt.text(data[i,0], data[i,1], str(labels[i]), color = colors[labels[i]],fontdict={'weight':'bold', 'size':9})
	plt.xlabel('First')
	plt.ylabel('Second')
	plt.savefig(name)
	plt.close()



def splitData(images, labels):
	image_list = [[] for _ in range(10)]
	for (image, label) in zip(images, labels):
		image_list[label].append(image)

	return image_list


def test():

	images, labels = load_data("training", SIZE_TRAINING)
	image_list = splitData(images, labels)

	index = 0	
	for item in image_list:
		# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
		c = 10
		pca = PCA(n_components=c)
		pca.fit(item)
		res = pca.transform(item)
		labels = [index for _ in range(len(item))]
		    
		draw("PCA label:{}".format(index), res, labels, "pca_{}.png".format(index),xlim=None, ylim=None)

		# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
		c = 10
		km = KMeans(n_clusters=c)
		km.fit(item)
		res = km.transform(item)    

		draw("k-means label:{}".format(index), res, labels, "kmeans_{}.png".format(index),xlim=None, ylim=None)

		# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
		c = 2
		tsne = TSNE(n_components=c, random_state=42)
		res = tsne.fit_transform(item)

		draw("t-SNE label:{}".format(index), res, labels, "tsne{}.png".format(index), xlim=None, ylim=None)


		index = index + 1



if __name__ == "__main__":
	test()


