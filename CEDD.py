# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:43:24 2022

@author: Anuj Patel
"""
import numpy as np
import cv2
from skimage.color import rgb2hsv
from ceddquant import ceddquant
from fuzzy10bin import fuzzy10bin
from fuzzy24bin import fuzzy24bin

def cedd(img):
    DescriptorVector = np.zeros(144)
    t0 = 14
    t1 = 0.68
    t2 = 0.98
    t3 = 0.98
    t = -1
    
    Compact = 0
     
    height = img.shape[0]
    width = img.shape[1]
    MeanRed = 0
    MeanGreen = 0
    MeanBlue = 0
    Edges = np.zeros(6)
    NeighborhoodArea1 = 0
    NeighborhoodArea2 = 0
    NeighborhoodArea3 = 0
    NeighborhoodArea4 = 0 
    Mask1 = 0
    Mask2 = 0
    Mask3 = 0
    Mask4 = 0
    Mask5 = 0
    Max = 0
    Fuzzy10BinResultTable = np.zeros(10)
    Fuzzy24BinResultTable = np.zeros(24)
    blocks=1600
    
    step_x = int(width/np.sqrt(blocks))
    step_y = int(height/np.sqrt(blocks))
    if step_x % 2 !=0 :
        step_x -= 1
    if step_y % 2 !=0 :
        step_y -= 1
        
    CororRed = np.zeros(step_x*step_y)
    CororGreen = np.zeros(step_x*step_y)
    CororBlue = np.zeros(step_x*step_y)
    
    ImageGridRed = img[:,:,0].T
    ImageGridGreen = img[:,:,1].T
    ImageGridBlue = img[:,:,2].T
    ImageGrid = 0.299*ImageGridRed + 0.587*ImageGridGreen + 0.114*ImageGridBlue
    
    TemoMAX_X = int(step_x*np.sqrt(blocks))
    TemoMAX_Y = int(step_y*np.sqrt(blocks))
    
    for y in range(0,TemoMAX_Y,step_y):
        for x in range(0,TemoMAX_X,step_x):
            MeanRed = 0
            MeanGreen = 0
            MeanBlue = 0
            NeighborhoodArea1 = 0
            NeighborhoodArea2 = 0
            NeighborhoodArea3 = 0
            NeighborhoodArea4 = 0
            Edges = np.ones(6)*(-1)
            TempSum = 0
            for i in range(y,y+step_y):
                for j in range(x,x+step_x):
                    CororRed[TempSum] = ImageGridRed[j,i]
                    CororGreen[TempSum] = ImageGridGreen[j,i]
                    CororBlue[TempSum] = ImageGridBlue[j,i]
                            
                    TempSum += 1
                    
                    if (j < (x+step_x/2)) and (i < (y+step_y/2)):
                        NeighborhoodArea1 += ImageGrid[j, i]
                    if (j >= (x+step_x/2)) and (i < (y+step_y/2)):
                        NeighborhoodArea2 += ImageGrid[j, i] 
                    if (j < (x+step_x/2)) and (i >= (y+step_y/2)):
                        NeighborhoodArea3 += ImageGrid[j, i] 
                    if (j >= (x+step_x/2)) and (i >= (y+step_y/2)):
                        NeighborhoodArea4 += ImageGrid[j, i] 
            
            NeighborhoodArea1 = int(np.fix(NeighborhoodArea1*4.0/(step_x*step_y)))
            NeighborhoodArea2 = int(np.fix(NeighborhoodArea2*4.0/(step_x*step_y)))
            NeighborhoodArea3 = int(np.fix(NeighborhoodArea3*4.0/(step_x*step_y)))
            NeighborhoodArea4 = int(np.fix(NeighborhoodArea4*4.0/(step_x*step_y)))
                
            Mask1 = abs(NeighborhoodArea1*2 + NeighborhoodArea2*(-2) + NeighborhoodArea3*(-2) + NeighborhoodArea4*2)
            Mask2 = abs(NeighborhoodArea1*1 + NeighborhoodArea2*1 + NeighborhoodArea3*(-1) + NeighborhoodArea4*(-1))
            Mask3 = abs(NeighborhoodArea1*1 + NeighborhoodArea2*(-1) + NeighborhoodArea3*1 + NeighborhoodArea4*(-1))
            Mask4 = abs(NeighborhoodArea1*np.sqrt(2) + NeighborhoodArea4*(-np.sqrt(2)) )
            Mask5 = abs(NeighborhoodArea2*np.sqrt(2) + NeighborhoodArea3*(-np.sqrt(2)) )              
            Max = max(Mask1,Mask2,Mask3,Mask4,Mask5)
            
            if Max>0:           
                Mask1 /= Max
                Mask2 /= Max
                Mask3 /= Max
                Mask4 /= Max
                Mask5 /= Max
            else:
                Mask1 = -np.inf
                Mask2 = -np.inf
                Mask3 = -np.inf
                Mask4 = -np.inf
                Mask4 = -np.inf    
                
            t=0    
            if Max<t0:
                Edges[0] = 0
                t=1            
            else:
                t=0
                if Mask1>t1:
                    Edges[t] = 1
                    t+=1
                if Mask2>t2:
                    Edges[t] = 2
                    t+=1
                if Mask3>t2:             
                    Edges[t] = 3
                    t+=1
                if Mask4>t3:
                    Edges[t] = 4
                    t+=1
                if Mask5>t3:
                    Edges[t] = 5
                    t+=1
                
            for i in range(step_x*step_y):
                MeanRed += CororRed[i]
                MeanGreen += CororGreen[i]
                MeanBlue += CororBlue[i]
            
            MeanRed = np.fix(MeanRed/(step_x*step_y))/255
            MeanGreen  = np.fix(MeanGreen /(step_x*step_y))/255
            MeanBlue  = np.fix(MeanBlue /(step_x*step_y))/255           
            HSV = rgb2hsv(np.array([MeanRed, MeanGreen, MeanBlue]))            
            HSV[0] = np.fix(HSV[0]*360)
            HSV[1] = np.fix(HSV[1]*255)
            HSV[2] = np.fix(HSV[2]*255)
    
            if Compact==0:
                Fuzzy10BinResultTable = fuzzy10bin(HSV[0], HSV[1], HSV[2], 2)
                Fuzzy24BinResultTable = fuzzy24bin(HSV[0], HSV[1], HSV[2], Fuzzy10BinResultTable, 2)
    
                for i in range(t):
                    for j in range(24):
                        if Fuzzy24BinResultTable[j]>0:
                            DescriptorVector[24 * int(Edges[i]) + j] += Fuzzy24BinResultTable[j]
            else:
                Fuzzy10BinResultTable = fuzzy10bin(HSV[0], HSV[1], HSV[2], 2)
                for i in range(t):
                    for j in range(10):
                        if Fuzzy10BinResultTable[j]>0:
                            DescriptorVector[10 * int(Edges[i]) + j] += Fuzzy10BinResultTable[j]
                            
    DescriptorVector = DescriptorVector/sum(DescriptorVector)
    DescriptorVector = ceddquant(DescriptorVector) 
    
    return DescriptorVector

img = cv2.imread("adobe (1).png")
print(cedd(img))
