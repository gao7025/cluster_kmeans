#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 17:48
# @Author  : gao
""""""

'''
确定聚类的最佳类数
'''


import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

class cal_clusters_num():
    def __init__(self, X, pathout):
        self.X = X
        self.pathout = pathout

    # 手肘法（elbow method）
    def kmeans_elbow(self):
        K = range(1, 10)
        meandistortions = []
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.X)
            meandistortions.append(
                sum(np.min(cdist(self.X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / self.X.shape[0])  # 最近邻距离（最小距离）的加和除以样本数
        plt.plot(K, meandistortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Average distortion')
        plt.title('Selecting k with the Elbow Method', fontsize=20)
        plt.savefig(self.pathout + '/elbow_cluster_diff_result.png', dpi=150)
        plt.show()

    # 层次聚类法
    def hierarchy_cal(self):
        dismatt = sch.distance.pdist(self.X, 'euclidean')  # 欧氏距离矩阵
        zz = sch.linkage(dismatt, method='average')
        pp = sch.dendrogram(zz)  # 树状图表示出来并保存
        plt.savefig(self.pathout + '/tree_cluster_diff_result.png', dpi=150)
        cluster_re = sch.fcluster(zz, t=1, criterion='inconsistent')
        print('hierarchy result:\n', cluster_re)
        return cluster_re