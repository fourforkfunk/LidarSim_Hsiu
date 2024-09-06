#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:54:44 2024

@author: hsiuweihsu


Hsiu-Wei, Hsu
School of Electrical and Computer Engineering, the University of Oklahoma, Norman, Oklahoma
Advanced Radar Research Center(ARRC),
Radar Innovation Laboratory (RIL),
E-mail: fourfork@gmail.com ; hsiuwei@ou.edu ;

"""
import os
import sys
import numpy as np
from scipy.special import erf
import argparse
from datetime import datetime, timedelta
import pandas as pd

#----------------------------------------------------------------------------------------------------
def comma_separated_list(value):
    return value.split(',')
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Displays radial velocity and SNR ...")
    parser.add_argument('-S',type=str, dest='scan')
    parser.add_argument('-AZ', default=['60','90'],type=comma_separated_list, dest='AZ')
    parser.add_argument('-fix', default=['90'],type=list, dest='fix')
    parser.add_argument('-AZ_reso', type=float, dest='reso')
    
    args = parser.parse_args()
    scan_type = args.scan
    AZ_range =  list(map(float, args.AZ )) # Make to float
    fixed_direction = list(map(float,  args.fix )) 
    AZ_reso = args.reso
    # print(args.fix)
    # print(fixed_direction)
    # sys.exit()
    # scan_type =  'PPI'
    if scan_type == 'PPI':
        print('PPI')
        AZ_need = np.arange(AZ_range[0],AZ_range[1]+AZ_reso,AZ_reso)
        AZ_array = [f"{x:07.3f}" for x in AZ_need ]
        Elv_array = [f"{x:07.3f}" for x in np.tile(fixed_direction,AZ_need.size)]
    
    elif scan_type == 'RHI':
        print('RHI'+str(fixed_direction))
    
       # AZ_range = [0,359]
       # fixed_direction = [70]
       # AZ_reso = 1
        AZ_need = np.arange(AZ_range[0],AZ_range[1]+AZ_reso,AZ_reso)
        AZ_array = [f"{x:07.3f}" for x in AZ_need ]
        Elv_array = [f"{x:07.3f}" for x in np.tile(fixed_direction,AZ_need.size)]
     
    elif scan_type == 'Pointing':
        print('Not yet.....')
        sys.exit()
        
    elif scan_type == 'Sur':    # Mulit-PPI 
        print('Not yet.....')
        sys.exit()
        
        
        
    # Output as two columns
    with open('Example_'+scan_type+'_test.txt', 'w') as file:
        # Write the header (optional)
        # file.write('Column1\tColumn2\n')
    
        # Iterate through both lists and write to file
        if scan_type == 'PPI':
            for item1, item2 in zip(AZ_array, Elv_array):
                file.write(f'{item1} {item2}\n')
        elif scan_type == 'RHI':
            for item1, item2 in zip(Elv_array, AZ_array):
                file.write(f'{item1} {item2}\n')

