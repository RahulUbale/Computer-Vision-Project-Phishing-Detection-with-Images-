# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 02:15:48 2022

@author: Rahul
"""

import sys
import cv2
import numpy as np
from PIL import Image
import argparse
import scipy.misc

def getbins(imgbk):
    M,N = imgbk.shape
    M = 2*np.ceil(M/2)
    N = 2*np.ceil(N/2)
    imgb = np.resize(imgbk,(int(M),int(N)))
    #cv2.imwrite("imgbk.png", imgbk)# Making block dimension divisible by 2
    bins = np.zeros((1,5)) # initialize Bin
    """Operations"""
    V = np.array([[1,-1],[1,-1]]) # Vertical  
    H = np.array([[1,1],[-1,-1]]) # Horizontal  
    D45 = np.array([[1.414,0],[0,-1.414]])# Diagonal 45  
    D135 = np.array([[0,1.414],[-1.414,0]]) # Diagonal 135  
    Isot = np.array([[2,-2],[-2,2]]) # Isotropic 
    T = 50 # threshold
    
    nobr = int(M/2) # loop limits
    nobc = int(N/2) # loop limits
    L = 0

    """loops of operating"""
    for _ in range(nobc):
        K = 0
        for _ in range(nobr):
            block = imgb[K:K+2, L:L+2] # Extracting 2x2 block
            pv = np.abs(np.sum(np.sum(block*V))) # apply operators
            ph = np.abs(np.sum(np.sum(block*H)))
            pd45 = np.abs(np.sum(np.sum(block*D45)))
            pd135 = np.abs(np.sum(np.sum(block*D135)))
            pisot = np.abs(np.sum(np.sum(block*Isot)))
            parray = [pv,ph,pd45,pd135,pisot]
            index = np.argmax(parray) # get the index of max value
            value = parray[index] # get the max value
            # print('value: '+str(value))
            if value >= T:
                bins[0,index]=bins[0,index]+1 # update bins values
            K = K+2
        L = L+2
    
    return bins 

def ehdimage(imgo):
    img1 = cv2.imread(imgo,0)
    r,c = np.shape(img1)
    img1 = Image.open(imgo)
    width, height = img1.size
    M = 4*np.ceil(r/4) 
    N = 4*np.ceil(c /4)
    img1 = img1.resize(((round(img1.width/4)*4),(round(img1.height/4)*4)))
    img1.save('resize.png')
    img1 = cv2.imread('resize.png',0)
    AllBins = np.zeros((17, 5))
    p = 1
    L = 0
    for _ in range(4):
        K = 0
        for _ in range(4):
            block = img1[K:K+int(M/4), L:L+int(N/4)]  #getting (M/4,N/4) block
            AllBins[p,:] = getbins(np.double(block)) 
            K = K + int(M/4)
            p = p + 1
        L = L + int(N/4)
    GlobalBin = np.mean(AllBins)  #getting global Bin
    AllBins[16,:]= np.round(GlobalBin)
    ehd = np.reshape(np.transpose(AllBins),[1,85])    
    ehd = ehd[0,-5:]

    return ehd


#a = ehdimage('phishIRIS_DL_Dataset\\phishIRIS_DL_Dataset\\train\\adobe\\adobe (15).png')  