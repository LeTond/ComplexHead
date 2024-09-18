 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import math
import sys

import matplotlib.pyplot as plt
import random as rand
import numpy as np
import nibabel as nib

from time import time
from pprint import pprint    


def read_nii(path_to_nii):
    # matplotlib.use('TkAgg')
    img = nib.load(path_to_nii)
    return img


def view_matrix(img):
    np.set_printoptions(threshold=sys.maxsize)
    return np.array(img.dataobj)


def view_img(img):
    plt.imshow(img)
    plt.show()



class InstancesFinder():
    def __init__(self, old_matrix, kernel):      
        self.kernel_sz = kernel
        self.old_matrix = old_matrix
        self.extraSymbol = 99
        self.symbols = list(range(2))
        self.layer = 3
        self.queue = []
        self.clusters = []
        self.min_distance = 1
        self.min_cluster_size = 1

        # Очередь из символов для поиска кластера
        self.directionsCluster = self.direction_cluster_genertor()

    def direction_cluster_genertor(self):
        
        directionsCluster = []
        
        for i in range(1, self.min_distance + 1):
        
            directionsCluster.append([0, i])
            directionsCluster.append([0, -i])
            directionsCluster.append([i, 0])
            directionsCluster.append([-i, 0])
            directionsCluster.append([-i, -i])
            directionsCluster.append([i, i])
            directionsCluster.append([-i, i])
            directionsCluster.append([i, -i])

        return directionsCluster

    # Поиск кластеров
    def findClusters(self):
     
        # Пустая матрица для пометки символов, которые уже участвовали в поиске кластеров
        markedSymbols = [[0 for i in range(self.kernel_sz)] for i in range(self.kernel_sz)] 
        
        # Перебираем все символы матрицы
        for i in range(self.kernel_sz):
            for j in range(self.kernel_sz):
                # Если символ - extra или помечен - пропускаем
                if (self.layer == self.extraSymbol or markedSymbols[i][j] == 3):
                    continue
                
                clusterData = {
                    'extras': [],
                    'coords': [],
                    'squares': []}
                
                # Добавляем текущий символ в очередь и помечаем его
                self.queue.append([i, j])
                markedSymbols[i][j] = self.layer

                # Пока в очереди что-то есть - перебираем соседние символы
                while (self.queue):
                    # Забираем символ из очереди
                    coords = self.queue.pop()   
                    # extra и обычные символы добавляем в разные массивы, тк у них разное поведение
                    
                    if (self.old_matrix[coords[0]][coords[1]] != self.extraSymbol):
                        clusterData['coords'].append(coords)
                    else:
                        clusterData['extras'].append(coords)      

                    # Перебираем все соседние символы
                    for direction in self.directionsCluster:
                        neighbourCoords = [coords[0] + direction[0], coords[1] + direction[1]]
                        try:
                            # Если соседний символ такой же или это extra (и не помечен) - добавляем его в очередь и помечаем
                            if ((self.old_matrix[neighbourCoords[0]][neighbourCoords[1]] == self.layer or 
                                self.old_matrix[neighbourCoords[0]][neighbourCoords[1]] == self.extraSymbol) and 
                            markedSymbols[neighbourCoords[0]][neighbourCoords[1]] == 0):
                            
                                self.queue.append(neighbourCoords)
                                markedSymbols[neighbourCoords[0]][neighbourCoords[1]] = 3      
                        except:
                                pass
                    
                # Берем только те кластеры, у которых длина больше 5 (учитывая extra)
                if (len(clusterData['coords']) + len(clusterData['extras']) >= (self.min_cluster_size + 1)):
                    clusterData['symbol'] = self.layer
                    self.clusters.append(clusterData)
                # Снимаем пометки с extra текущего кластера, тк они могут быть частью и других кластеров
                for coords in clusterData['extras']:
                    markedSymbols[coords[0]][coords[1]] = 0

        return self.clusters


    def iteration(self):
        ...

    def new_instance_matrix(self):

        main_matrix = []
        new_matrix = np.copy(self.new_matrix())

        shp_old = new_matrix.shape

        for lyr in np.unique(new_matrix):
            matrix = np.copy(new_matrix)

            if lyr < 3:
                matrix[matrix != lyr] = 0
                main_matrix.append(matrix)
            elif lyr >= 13: 
                matrix[matrix != lyr] = 0
                matrix[matrix == lyr] = 3
                main_matrix.append(matrix)

        main_matrix = np.array(main_matrix).transpose(2, 1, 0)
        
        shp_new =  main_matrix.shape
        print(f'Matrix shape was changed from {shp_old} to {shp_new}')

        return main_matrix

    def new_matrix(self):

        new_matrix = np.copy(self.old_matrix)
        cluster = self.findClusters()

        for i in range(len(cluster[:])):
            for j in range(1, len(cluster[i]['coords'])):

                new_layer = self.layer + i + 10 # i - from 0 to count of found classes
                coord_ = cluster[i]['coords'][j]         
                new_matrix[coord_[0]][coord_[1]] = new_layer
                
        return new_matrix

    def threshold_matrix(self):

        new_matrix = np.copy(self.old_matrix)
        cluster = self.findClusters()

        for i in range(len(cluster[:])):
            if len(cluster[i]['coords']) == 2:
                for j in range(1, len(cluster[i]['coords'])):
                
                    new_layer = 2 # i - from 0 to count of found classes
                    coord_ = cluster[i]['coords'][j]         
                    new_matrix[coord_[0]][coord_[1]] = new_layer
                    
        return new_matrix


if __name__ == "__main__":
    old_matrix = [
                [0, 3, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 3, 3, 3, 0, 1],
                [1, 0, 1, 3, 3, 3, 1, 0, 1],
                [1, 1, 3, 0, 1, 1, 0, 1, 3],
                [1, 1, 0, 1, 3, 3, 3, 1, 3],
                [1, 1, 0, 1, 1, 3, 3, 1, 0],
                [0, 3, 1, 0, 1, 0, 0, 0, 1],
                [0, 3, 3, 2, 2, 0, 3, 1, 1],
                [0, 0, 2, 2, 2, 1, 1, 1, 0]
                ]

    # generate it to npy

    image_matrix = view_matrix(read_nii("./Sub03.nii"))
    image_matrix = image_matrix[:,:,4]
    # new_instance_matrix = InstancesFinder(image_matrix, kernel = 144).new_instance_matrix()
    # new_instance_matrix = InstancesFinder(image_matrix, kernel = 144).new_matrix()
    
    new_matrix = InstancesFinder(old_matrix, kernel = 9).new_matrix()
    print(new_matrix)
    # view_img(new_instance_matrix)


    threshold_matrix = InstancesFinder(old_matrix, kernel = 9).threshold_matrix()
    print(threshold_matrix)



    # fromn  (4, 144, 192) to  (144, 192, 4)




