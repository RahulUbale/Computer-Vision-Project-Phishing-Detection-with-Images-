# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:50:52 2022

@author: Anuj Patel
"""
import numpy as np

def fuzzy24bin(hue, saturation, value, Fuzzy10binHist, method):
    histogram = np.zeros(24)
    ResultsTable = np.zeros(3)
    
    SaturationMembershipValues = [0, 0, 68, 188, 68, 188, 255, 255]
    ValueMembershipValues = [0, 0, 68, 188, 68, 188, 255, 255]
    
    SaturationActivation = np.zeros(2)
    ValueActivation = np.zeros(2)
    Temp = 0
    
    TempSaturation = 0
    TempValue = 0
    for i in range(0,8,4):
        if saturation >= SaturationMembershipValues[i+1] and saturation <= SaturationMembershipValues[i+2]:
            SaturationActivation[TempSaturation] = 1
        if saturation >= SaturationMembershipValues[i] and saturation < SaturationMembershipValues[i+1]:
            SaturationActivation[TempSaturation] = (saturation - SaturationMembershipValues[i]) / (SaturationMembershipValues[i+1] - SaturationMembershipValues[i])
        if saturation > SaturationMembershipValues[i+2] and saturation < SaturationMembershipValues[i+3]:
            SaturationActivation[TempSaturation] = (saturation - SaturationMembershipValues[i+2]) / (SaturationMembershipValues[i+2] - SaturationMembershipValues[i+3]) + 1
        TempSaturation += 1
    
        if value >= ValueMembershipValues[i+1] and value <= ValueMembershipValues[i+2]:
            ValueActivation[TempValue] = 1
        if value >= ValueMembershipValues[i] and value < ValueMembershipValues[i+1]:
            ValueActivation[TempValue] = (value - ValueMembershipValues[i]) / (ValueMembershipValues[i+1] - ValueMembershipValues[i])
        if value > ValueMembershipValues[i+2] and value < ValueMembershipValues[i+3]:
            ValueActivation[TempValue] = (value - ValueMembershipValues[i+2]) / (ValueMembershipValues[i+2] - ValueMembershipValues[i+3]) + 1  
        TempValue += 1
    
    for i in range(3,10):
        Temp += Fuzzy10binHist[i]
        
    Fuzzy24BinRulesDefinition = np.array([[1,1,1],
                              [0,0,2],                   
                              [0,1,0],
                              [1,0,2]])
                                         
    if Temp>0:
        for i in range(len(Fuzzy24BinRulesDefinition)):
            if (SaturationActivation[Fuzzy24BinRulesDefinition[i,0]] > 0) and (ValueActivation[Fuzzy24BinRulesDefinition[i,1]]>0):
                RuleActivation = Fuzzy24BinRulesDefinition[i,2]
                Minimum = min( SaturationActivation[Fuzzy24BinRulesDefinition[i,0]], ValueActivation[Fuzzy24BinRulesDefinition[i,1]] )
                ResultsTable[RuleActivation] += Minimum
    
    for i in range(3):
        histogram[i] += Fuzzy10binHist[i]       
    for i in range(3,10):
        histogram[(i-2)*3] += Fuzzy10binHist[i]*ResultsTable[0]
        histogram[(i-2)*3+1] += Fuzzy10binHist[i]*ResultsTable[1]
        histogram[(i-2)*3+2] += Fuzzy10binHist[i]*ResultsTable[2]
        
    return histogram