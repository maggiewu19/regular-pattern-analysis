import cv2 
import math 
import os 
import copy 
import json 
import operator

import numpy as np 
import itertools as it 
import matplotlib.pyplot as plt 
import sklearn.preprocessing as sp 
import pickle as pkl 

def save_pickle(fname, data):
    '''
    Save data into pickle file with fname 

    Input:  fname (str) - pickle file
            data (data type) - data structure 
    Output: none 
    '''
    with open(fname, 'wb') as pickle_file: 
        pkl.dump(data, pickle_file, protocol=pkl.HIGHEST_PROTOCOL)

def load_pickle(fname, ptype): 
    '''
    Load pickle if pickle file exists 

    Input:  fname (str) - pickle file 
            ptype (data structure) - type of the data 
    Output: data (data structure) - data structure of type ptype 
    '''
    if os.path.isfile(fname): 
        with open(fname, 'rb') as pickle_file: 
            data = pkl.load(pickle_file)
        return data 
    return ptype

def load_json(fname):
    '''
    Load json if json file exists 

    Input:  fname (str) - json file 
    Output: data (dict) - dictionary of json file 
    '''
    with open(fname) as json_file: 
        data = json.load(json_file)
    return data 

def euclidean(x1, y1, x2, y2): 
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def transform_coord(x, y, homography):
    '''
    Transform point by homography 

    Input:  pt (float) - coordinates
            homography (matrix) - homography 
    Output: pt (float) - new coordinates 
    '''
    dotProd = np.dot(homography, np.array([x, y, 1])) 
    newX, newY = dotProd[0]/dotProd[2], dotProd[1]/dotProd[2]
    return float(newX), float(newY)

def distance_error(points1, points2, imageUnit=48):
    n = len(points1)
    distances = [np.linalg.norm(np.array(points1[i])-np.array(points2[i])) for i in range(n)]

    meanDist = sum(distances)/len(distances)
    errorCount = sum([1 if d >= imageUnit else 0 for d in distances])

    return meanDist, errorCount 