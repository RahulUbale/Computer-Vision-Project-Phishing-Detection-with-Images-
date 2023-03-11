# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:08:52 2022

@author: Anuj Patel
"""
import numpy as np

def fuzzy10bin(hue, saturation, value, method):
    histogram = np.zeros(10)
    
    HueMembershipValues = [0,0,5, 10,5,10,35,50,35,50,70, 85,70,85,150, 165, 150,165,195, 205,195,205,265, 280,265,280,315, 330, 315,330,360,360]
    SaturationMembershipValues = [0,0,10, 75,10,75,255,255]
    ValueMembershipValues = [0,0,10,75,10,75,180,220,180,220,255,255]
    
    HueActivation = np.zeros(8)
    SaturationActivation = np.zeros(2)
    ValueActivation = np.zeros(3)
    
    TempHue = 0
    TempSaturation = 0
    TempValue = 0
    for i in range(0,32,4):
        if i<8:
            if saturation >= SaturationMembershipValues[i+1] and saturation <= SaturationMembershipValues[i+2]:
                SaturationActivation[TempSaturation] = 1
            if saturation >= SaturationMembershipValues[i] and saturation < SaturationMembershipValues[i+1]:
                SaturationActivation[TempSaturation] = (saturation - SaturationMembershipValues[i]) / (SaturationMembershipValues[i+1] - SaturationMembershipValues[i])
            if saturation > SaturationMembershipValues[i+2] and saturation < SaturationMembershipValues[i+3]:
                SaturationActivation[TempSaturation] = (saturation - SaturationMembershipValues[i+2]) / (SaturationMembershipValues[i+2] - SaturationMembershipValues[i+3]) + 1
            TempSaturation += 1
        
        if i<12:
            if value >= ValueMembershipValues[i+1] and value <= ValueMembershipValues[i+2]:
                ValueActivation[(TempValue) ] = 1
            if value >= ValueMembershipValues[i] and value < ValueMembershipValues[i+1]:
                ValueActivation[TempValue] = (value - ValueMembershipValues[i]) / (ValueMembershipValues[i+1] - ValueMembershipValues[i])
            if value > ValueMembershipValues[i+2] and value < ValueMembershipValues[i+3]:
                ValueActivation[TempValue] = (value - ValueMembershipValues[i+2]) / (ValueMembershipValues[i+2] - ValueMembershipValues[i+3]) + 1  
            TempValue += 1
            
        if hue >= HueMembershipValues[i+1] and hue <= HueMembershipValues[i+2]:
            HueActivation[TempHue] = 1
        if hue >= HueMembershipValues[i] and hue < HueMembershipValues[i+1]:
            HueActivation[TempHue] = (hue - HueMembershipValues[i]) / (HueMembershipValues[i+1] - HueMembershipValues[i])
        if hue > HueMembershipValues[i+2] and hue < HueMembershipValues[i+3]:
            HueActivation[TempHue] = (hue - HueMembershipValues[i+2]) / (HueMembershipValues[i+2] - HueMembershipValues[i+3]) + 1
        TempHue += 1
    
    Fuzzy10BinRulesDefinition = np.array([[0,0,0,2],[0,1,0,2],
                              [0,0,2,0],
                              [0,0,1,1],
                              [1,0,0,2],                        
                              [1,1,0,2],
                              [1,0,2,0],
                              [1,0,1,1],
                              [2,0,0,2],                        
                              [2,1,0,2],
                              [2,0,2,0],
                              [2,0,1,1],
                              [3,0,0,2],                       
                              [3,1,0,2],
                              [3,0,2,0],
                              [3,0,1,1],
                              [4,0,0,2],                        
                              [4,1,0,2],
                              [4,0,2,0],
                              [4,0,1,1],
                              [5,0,0,2],                        
                              [5,1,0,2],
                              [5,0,2,0],
                              [5,0,1,1],
                              [6,0,0,2],                        
                              [6,1,0,2],
                              [6,0,2,0],
                              [6,0,1,1],
                              [7,0,0,2],                        
                              [7,1,0,2],
                              [7,0,2,0],
                              [7,0,1,1],
                              [0,1,1,3],
                              [0,1,2,3],                   
                              [1,1,1,4],
                              [1,1,2,4],
                              [2,1,1,5],
                              [2,1,2,5],
                              [3,1,1,6],
                              [3,1,2,6],
                              [4,1,1,7],
                              [4,1,2,7],
                              [5,1,1,8],
                              [5,1,2,8],
                              [6,1,1,9],
                              [6,1,2,9],
                              [7,1,1,3],
                              [7,1,2,3]])
    
    if method==2:
        for i in range(48):
            if HueActivation[Fuzzy10BinRulesDefinition[i,0]]>0 and SaturationActivation[Fuzzy10BinRulesDefinition[i,1]]>0 and ValueActivation[Fuzzy10BinRulesDefinition[i,2]]>0:
                RuleActivation = Fuzzy10BinRulesDefinition[i,3]
                Minimum = min( HueActivation[Fuzzy10BinRulesDefinition[i,0]] , SaturationActivation[Fuzzy10BinRulesDefinition[i,1]] , ValueActivation[Fuzzy10BinRulesDefinition[i,2]] )
                
                histogram[RuleActivation] += Minimum   
    
    return histogram