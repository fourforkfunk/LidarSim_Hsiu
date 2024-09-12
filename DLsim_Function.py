#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:54:17 2024

@author: Josh, hsiuweihsu

Josh...


Hsiu-Wei, Hsu
School of Electrical and Computer Engineering, the University of Oklahoma, Norman, Oklahoma
Advanced Radar Research Center(ARRC),
Radar Innovation Laboratory (RIL),
E-mail: fourfork@gmail.com ; hsiuwei@ou.edu ; 

"""
import os
import sys
import numpy as np
import glob
import pyproj
import time
import struct
import xarray as xr
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from scipy.special import erf
# from argparse import ArgumentParser
from datetime import datetime, timedelta


class Read_file:
    # a = 1
    # b = 2
    def __init__(self):     

        pass
        # self.name = 'name'
        # self.food = 'food'
            
    def read_namelist(filename):
        
        # This large structure will hold all of the namelist option. We are
        # predefining it so that default values will be used if they are not listed
        # in the namelist. The exception to this is if the user does not specify 
        # the model. In this scenerio the success flag will terminate the program.
        
        namelist = ({'success':0,
                     'model':0,                 # Type of model used for the simulation. 1-WRF, 2-FastEddy, 3-NCAR LES
                     'model_frequency':0.0 ,       # Frequency of model output used for lidar simulation in seconds
                     'model_timestep':0.025,    # Model time step (needed for FE and CM1)
                     'model_dir':'None',        # Directory with the model data
                     'model_prefix':'None',     # Prefix for the model data files
                     'outfile':'None',
                     'append':0,
                     'clobber':0,
                     'instantaneous_scan':1,    # 0-Collect data realistically (need high temporal resolution output), 1-Collect data instantaneously at each model time
                     'number_scans':1,          # Number of scan files to read in
                     'scan_file1':'None',
                     'cc1':0,                  # 1-Means continuously cycle scan 1. Ignored if instantaneous_scan is 1.
                     'repeat1':0.0,               # The repeat time of scan 1. Ignored if cc1 is 1 or instantaneous_scan is 1.
                     'dbs1':0,                    # 1 means scan 1 is a DBS scan with gates specified by height agl
                     'scan_file2':'None',
                     'cc2':0,                    # 1-Means continuously cycle scan 2. Ignored if instantaneous_scan is 1.
                     'repeat2':0.0,                # The repeat time of scan 2. Ignored if cc2 is 1 or instantaneous_scan is 1.
                     'dbs2':0,                   # 1 means scan 1 is a DBS scan with gates specified by height agl
                     'scan_file3':'None',
                     'cc3':0,                    # 1-Means continuously cycle scan 3. Ignored if instantaneous_scan is 1.
                     'repeat3':0.0,                # The repeat time of scan 3. Ignored if cc3 is 1 or instantaneous_scan is 1.
                     'dbs3':0,                   # 1 means scan 1 is a DBS scan with gates specified by height agl
                     'scan_file4':'None',
                     'cc4':0,                    # 1-Means continuously cycle scan 4. Ignored if instantaneous_scan 1.
                     'repeat4':0.0,                # The repeat time of scan 4. Ignored if cc4 is 1 or instantaneous_scan is 1.
                     'dbs4':0,                   # 1 means scan 1 is a DBS scan with gates specified by height agl
                     'stare_length':0.0,            # Length of vertical stare in between scheduled scans in seconds.
                     'motor_az_speed':36.0,       # In degrees/sec (ignored if instantaneous)
                     'motor_el_speed':36.0,       # In degrees/sec (ignored if instantaneous)
                     'scan1_az_speed':0,          # Azimuth speed for scan 1, 0 is same as general, -1 CSM with 1s ray accumulation time
                     'scan1_el_speed':0,          # Elevation speed for scan 1, same as above
                     'scan2_az_speed':0,          # Azimuth speed for scan 2, same as above
                     'scan2_el_speed':0,          # Elevation speed for scan 2, same as above
                     'scan3_az_speed':0,          # Azimuth speed for scan 3, same as above
                     'scan3_el_speed':0,          # Elevation speed for scan 3, same as above
                     'scan4_az_speed':0,          # Azimuth speed for scan 4, same as above
                     'scan4_el_speed':0,          # Elevation speed for scan 4, same as above
                     'ray_time':1.0,                # Ray accumulation time in seconds
                     'pulse_width':150.0,         # length of pulse in ns
                     'gate_width':200.0,          # length of range gates in ns
                     'maximum_range':5.0,         # maximum range of the lidar in km
                     'dbs_start_height':40,       # Start height of DBS scan
                     'dbs_end_height':480,        # End height of DBS scans
                     'dbs_spacing':20,            # Vertical spacing of dbs points
                     'sample_resolution':10.,      # sample resolution of the model along the lidar beam in m.
                     'nyquist_velocity':19.4,    # Nyquist velocity of the lidar
                     'coordinate_type':1,        # 1-Lat/Lon, 2-x,y,z
                     'lidar_lat':35.3,           # latitude of simulated lidar in degrees (ignored if coordinate_type is 2)
                     'lidar_lon':-97.8,          # longtitude of simulated lidar in degrees (ignored if coordinate_type is 2)
                     'lidar_x':5000.0,             # x position of simulated lidar in default units (ignored if coordinate_type is 1)
                     'lidar_y':5000.0,             # y position of simulated lidar in default units  (ignored if coordinate_type is 1)
                     'lidar_alt':300.0,            # height of the simulated lidar (m above sea level)
                     'use_calendar':1,           # If 1 then the start and end times are defined by calendar If 0 they are in model integration time
                     'start_year':0,           # Ignored if use_calendar is 0
                     'start_month':0,           # Ignored if use_calendar is 0
                     'start_day':0,             # Ignored if use_calendar is 0
                     'start_hour':0,       # Ignored if use_calendar is 0
                     'start_min':0,           # Ignored if use_calendar is 0
                     'start_sec':0,           # Ignored if use_calendar is 0
                     'end_year':0,             #Ignored is use_calendar is 0
                     'end_month':0,         # Ignored if use_calendar is 0
                     'end_day':0,           # Ignored if use_calendar is 0
                     'end_hour':0,          # Ignored if use_calendar is 0
                     'end_min':0,         # Ignored if use_calendar is 0
                     'end_sec':0,           # Ignored if use_calendar is 0
                     'start_time':0.0,              # Start time of the lidar simulation (Ignored if used calendar is 1)
                     'end_time':86400.0,            # End time of the lidar simulation
                     'ncar_les_nscl':21,    # Number of scalars in NCAR LES output
                     'clouds':0,            # Extinguish beam for clouds (Only works with WRF and CM1)  
                     'turb':0,              # Add variance to the radial velocities along beam from subgrid turbulence (Only works with CM1)
                     'sim_signal':0,        # Simulate lidar signal loss based on signal climatology
                     'signal_climo_file':'None', # Path to the lidar signal climotology file
                     'points_per_gate':20,  # Number of lidar samples per range gate
                     'num_pulses':10000,    # Number of pulses per gate
                     'umove':0.0,             # East west grid velocity in m/s
                     'vmove':0.0,
                     'psudeo_pulses':1000 ,    # In number (ignored if Stared-stop mode) 
                     'prf':5000,
                     'scan_type':0,# In Hz (ignored if Stared-stop mode)
                     'scan_type_direction':'None'# In Hz (ignored if Stared-stop mode) 

                     }             # North south grid velocity in m/s 
                     )
        
        # Read in the file all at once
        
        if os.path.exists(filename):
            print('Reading the namelist: ' + filename)
            
            try:
                inputt = np.genfromtxt(filename, dtype=str, comments ='#',delimiter='=', autostrip=True)
            except:
                print ('ERROR: There was a problem reading the namelist')
                return namelist
        
        else:
            print('ERROR: The namelist file ' + namelist + ' does not exist')
            return namelist
        
        if len(inputt) == 0:
            print('ERROR: There were not valid lines found in the namelist')
            return namelist
        
        # This is where the values in the namelist dictionary are changed
        
        nfound = 1
        for key in namelist.keys():
            if key != 'success':
                nfound += 1
                foo = np.where(key == inputt[:,0])[0]
                if len(foo) > 1:
                    print('ERROR: There were multiple lines with the same key in the namelist: ' + key)
                    return namelist
                
                elif len(foo) == 1:
                    namelist[key] = type(namelist[key])(inputt[foo,1][0])
                
                else:
                    nfound -= 1
        
        scan_speeds = np.zeros((5,2))
        scan_speeds[0,0] = namelist['motor_az_speed']
        scan_speeds[0,1] = namelist['motor_el_speed']
        
        if namelist['scan1_az_speed'] == 0:
            scan_speeds[1,0] = namelist['motor_az_speed']
        else:
            scan_speeds[1,0] = namelist['scan1_az_speed']
            
        if namelist['scan1_el_speed'] == 0:
            scan_speeds[1,1] = namelist['motor_el_speed']
        else:
            scan_speeds[1,1] = namelist['scan1_el_speed']
        
        if namelist['scan2_az_speed'] == 0:
            scan_speeds[2,0] = namelist['motor_az_speed']
        else:
            scan_speeds[2,0] = namelist['scan2_az_speed']
            
        if namelist['scan2_el_speed'] == 0:
            scan_speeds[2,1] = namelist['motor_el_speed']
        else:
            scan_speeds[2,1] = namelist['scan2_el_speed']
            
        if namelist['scan3_az_speed'] == 0:
            scan_speeds[3,0] = namelist['motor_az_speed']
        else:
            scan_speeds[3,0] = namelist['scan3_az_speed']
            
        if namelist['scan3_el_speed'] == 0:
            scan_speeds[3,1] = namelist['motor_el_speed']
        else:
            scan_speeds[3,1] = namelist['scan3_el_speed']
            
        if namelist['scan4_az_speed'] == 0:
            scan_speeds[4,0] = namelist['motor_az_speed']
        else:
            scan_speeds[4,0] = namelist['scan4_az_speed']
            
        if namelist['scan4_el_speed'] == 0:
            scan_speeds[4,1] = namelist['motor_el_speed']
        else:
            scan_speeds[4,1] = namelist['scan4_el_speed']
        
        # A quick check here to make sure that the namelist parameters are valid for the settings
        
        if namelist['model'] == 0:
            print('ERROR: The model used for the simulation needs to be specified!')
            return namelist
        
        if namelist['model_frequency'] == 0:
            print('ERROR: Need to specify the frequency of the model output!')
            return namelist
        
        if namelist['model_dir'] == 'None':
            print('ERROR: Need to specify the model output directory')
        
        if ((namelist['instantaneous_scan'] == 0) & (namelist['model_frequency'] > 5)):
            print('##################################################################################')
            print('# WARNING: Model output is not frequent enough for realistic scanning simulation #')
            print('#          Recommend changing instantaneous_scan to 1.                         #')
            print('##################################################################################')
        
        # We are making an array here to keep track of the cc'ed scans as well as 
        # checking to make sure that non cc'ed scan do not have repeat intervals of 0 seconds.
        cced = np.zeros(namelist['number_scans'])
        repeats = np.zeros(namelist['number_scans'])
        for i in range(namelist['number_scans']):
            if ((namelist['cc' + str(i+1)] == 0) & (namelist['repeat' + str(i+1)] == 0.0)):
                print('WARNING: Repeat interval for scan file ' + str(i+1) + ' is 0.0 (essentially continuous),')
                print('         but cc' + str(i+1) + ' is 0. Setting cc' + str(i+1) + ' to 1.')
                namelist['cc' + str(i+1)] = 1
                
            if namelist['cc' + str(i+1)] == 1:
                cced[i] = 1
                
            repeats[i] = namelist['repeat' + str(i+1)]
        namelist['cced'] = cced
        namelist['repeats'] = repeats      
          
        if namelist['use_calendar'] == 1:
            if ((namelist['start_year'] == 0) | (namelist['start_month'] == 0) | (namelist['start_day'] == 0) |
                (namelist['end_year'] == 0) | (namelist['end_month'] == 0) | (namelist['end_day'] == 0)):
                    print('ERROR: If use_calendar = 1, then start_year, start_month, start_day, end_year, end_month, and end_year cannot be 0.')
                    return namelist
        
        if (namelist['number_scans'] < 1) | (namelist['number_scans'] > 4):
            print('ERROR: Number of scans must be 1-4 for now.....')
            return namelist
        
        namelist['success'] = 1
        return namelist,scan_speeds
    
    ##############################################################################
    # Get the data you need from a WRF run
    ##############################################################################

    def get_wrf_data(x,y,z,lidarx,lidary,lidarz,file,cloud,xx,yy,transform):
        
        f = Dataset(file)
    
        if xx is None:
            xx, yy = np.meshgrid(np.arange(f.dimensions['west_east'].size) * f.DX, np.arange(f.dimensions['south_north'].size) * f.DY)
        
        zz = (f['PH'][0] + f['PHB'][0])/9.81
        zz = (zz[1:] + zz[:-1])/2.
        
        zz = zz.T
        xx = xx.T
        yy = yy.T
        
        # Interpolation with WRF is hard because of the terrain following grid.
        # Triangulation methods are too slow with large datasets, so we are using
        # a tensor-product interpolation. This generates extra data (i.e more memory intensive),
        # but is significantly faster
        
        # First check if max x is less than the minimum of the grid or if min x is greater than max x
        # and if so just return a bunch of nans because the beam is never in the domain
        if (np.max(x) < np.min(xx[:,0])) or (np.min(x) > np.max(xx[:,0])):
            return np.ones(len(x))*np.nan
        
        # Do the same thing for y
        if (np.max(y) < np.min(yy[0])) or (np.min(y) > np.max(yy[0])):
            return np.ones(len(x))*np.nan
        
        # First chunk the data for the region we need
        
        if np.max(x) > np.max(xx[:,0]):
            ixmax = f.dimensions['west_east'].size-1
            if np.min(x) < np.min(xx[:,0]):
                ixmin = 0
            else:
                ixmin = np.where((np.min(x) > xx[:,0]))[0][-1]
        elif np.min(x) < np.min(xx[:,0]):
            ixmin = 0
            ixmax = np.where((np.max(x) <= xx[:,0]))[0][0]
        else:
            ixmin, ixmax = np.where((np.min(x) > xx[:,0]))[0][-1], np.where((np.max(x) <= xx[:,0]))[0][0]
        
        
        if np.max(y) > np.max(yy[0]):
            iymax = f.dimensions['south_north'].size-1
            if np.min(y) < np.min(yy[0]):
                iymin = 0
            else:
                iymin = np.where((np.min(y) > yy[0]))[0][-1]
        elif np.min(y) < np.min(yy[0]):
            iymin = 0
            iymax = np.where((np.max(y) <= yy[0]))[0][0]
        else:
            iymin, iymax = np.where((np.min(y) > yy[0]))[0][-1], np.where((np.max(y) <= yy[0]))[0][0]
        
        if np.max(z) > np.max(np.min(np.min(zz,axis=0),axis=0)):
            izmax = zz.shape[0]-1
        else:
            izmax = np.where(np.max(z) < np.min(np.min(zz,axis=0),axis=0))[0][0]
            
        u = (f['U'][0,:,:,1:] + f['U'][0,:,:,:-1])/2.
        u = u[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        v = (f['V'][0,:,1:,:] + f['V'][0,:,:-1,:])/2.
        v = v[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        if transform is not None:
            sinalpha = f['SINALPHA'][0,iymin:iymax+1,ixmin:ixmax+1].T
            cosalpha = f['COSALPHA'][0,iymin:iymax+1,ixmin:ixmax+1].T
            u_tmp = u*cosalpha[:,:,None] - v*sinalpha[:,:,None]
            v_tmp = v*cosalpha[:,:,None] + u*sinalpha[:,:,None]
            u = np.copy(u_tmp)
            v = np.copy(v_tmp)
    
        w = (f.variables['W'][0,1:,:,:] + f.variables['W'][0,:-1,:,:])/2.
        w = w[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        
        zzz = zz[ixmin:ixmax+1,iymin:iymax+1,:izmax+1]
        
        qi = (x,y)
        q = (xx[ixmin:ixmax+1,0],yy[0,iymin:iymax+1])
        
        # Although inefficient I am braking this up into 4 different loops because
        # I am concerned about memory usage for a long range lidar
        
        idx = np.identity(len(x))*np.arange(1,len(x)+1)
        
        for j in range(2):
            u = interp1d(q[j],u,axis=j,bounds_error=False)(qi[j])
            idx = np.delete(idx,np.where(np.isnan(u))[j],axis=j)
            u = np.delete(u,np.where(np.isnan(u))[j],axis=j)
        
        foo = np.where(idx != 0)
              
        u = u[foo[0],foo[1],:]
        
        for j in range(2):
            v = interp1d(q[j],v,axis=j,bounds_error=False)(qi[j])
            v = np.delete(v,np.where(np.isnan(v))[j],axis=j)
        
        v = v[foo[0],foo[1],:]
        
        for j in range(2):
            w = interp1d(q[j],w,axis=j,bounds_error=False)(qi[j])
            w = np.delete(w,np.where(np.isnan(w))[j],axis=j)
        
        w = w[foo[0],foo[1],:]
        
        
        for j in range(2):
            zzz = interp1d(q[j],zzz,axis=j,bounds_error=False)(qi[j])
            zzz = np.delete(zzz,np.where(np.isnan(zzz))[j],axis=j)
        
        zzz = zzz[foo[0],foo[1],:]
        
        if cloud == 1:
            qtotal  = (f['QCLOUD'][0,:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T +
                       f['QRAIN'][0,:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T)
            
            for j in range(2):
                qtotal = interp1d(q[j],qtotal,axis=j,bounds_error=False)(qi[j])
                qtotal = np.delete(qtotal,np.where(np.isnan(qtotal))[j],axis=j)
            
            qtotal = qtotal[foo[0],foo[1],:]
            
        
        idx = (idx[foo] - 1).astype(int)
    
        vr = []
        cloudfree = True
        for i in range(len(x)):
            foo = np.where(i == idx)[0]
            
            if len(foo) == 0:
                vr.append(np.nan)
            else:
                if cloud == 1:
                    temp_q = np.interp(z[i],zzz[foo[0]],qtotal[foo[0]],left=np.nan,right=np.nan)
                    if temp_q > 0.00001:
                        cloudfree = False
                        
                if cloudfree:
                    temp_u = np.interp(z[i],zzz[foo[0]],u[foo[0]],left=np.nan,right=np.nan)
                    temp_v = np.interp(z[i],zzz[foo[0]],v[foo[0]],left=np.nan,right=np.nan)
                    temp_w = np.interp(z[i],zzz[foo[0]],w[foo[0]],left=np.nan,right=np.nan)
            
                    vr.append(((x[i]-lidarx)*temp_u + (y[i]-lidary)*temp_v + (z[i]-lidarz)*temp_w)/np.sqrt((x[i]-lidarx)**2 + (y[i]-lidary)**2 + (z[i]-lidarz)**2))
                else:
                    vr.append(np.nan)
    
        return np.array(vr) 
    
    
    ##############################################################################
    # Get the data you need from a FastEddy run
    ##############################################################################
    
    def get_fasteddy_data(x,y,z,lidarx,lidary,lidarz,file):
        f = Dataset(file,'r')
        
        zz = f['zPos'][0]
        xx = f['xPos'][0,0]
        yy = f['yPos'][0,0]
        
        zz = zz.T
        xx = xx.T
        yy = yy.T
        
        # First check if max x is less than the minimum of the grid or if min x is greater than max x
        # and if so just return a bunch of nans because the beam is never in the domain
        if (np.max(x) < np.min(xx[:,0])) or (np.min(x) > np.max(xx[:,0])):
            return np.ones(len(x))*np.nan
        
        # Do the same thing for y
        if (np.max(y) < np.min(yy[0])) or (np.min(y) > np.max(yy[0])):
            return np.ones(len(x))*np.nan
        
        if np.max(x) > np.max(xx[:,0]):
            ixmax = xx.shape[0]-1
            if np.min(x) < np.min(xx[:,0]):
                ixmin = 0
            else:
                ixmin = np.where((np.min(x) > xx[:,0]))[0][-1]
        elif np.min(x) < np.min(xx[:,0]):
            ixmin = 0
            ixmax = np.where((np.max(x) <= xx[:,0]))[0][0]
        else:
            ixmin, ixmax = np.where((np.min(x) > xx[:,0]))[0][-1], np.where((np.max(x) <= xx[:,0]))[0][0]
        
        
        if np.max(y) > np.max(yy[0]):
            iymax = yy.shape[1]-1
            if np.min(y) < np.min(yy[0]):
                iymin = 0
            else:
                iymin = np.where((np.min(y) > yy[0]))[0][-1]
        elif np.min(y) < np.min(yy[0]):
            iymin = 0
            iymax = np.where((np.max(y) <= yy[0]))[0][0]
        else:
            iymin, iymax = np.where((np.min(y) > yy[0]))[0][-1], np.where((np.max(y) <= yy[0]))[0][0]
        
        if np.max(z) > np.max(np.min(np.min(zz,axis=0),axis=0)):
            izmax = zz.shape[0]-1
        else:
            izmax = np.where(np.max(z) < np.min(np.min(zz,axis=0),axis=0))[0][0]
        
        u = f['u'][0]
        
        u = u[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        v = f['v'][0]
        
        v = v[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        w = f['w'][0]
        
        w = w[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        ground = f['topoPos'][0]
        
        f.close()
        ground = ground[iymin:iymax+1,ixmin:ixmax+1].T
        
        zzz = zz[ixmin:ixmax+1,iymin:iymax+1,:izmax+1]
        
        qi = (x,y)
        q = (xx[ixmin:ixmax+1,0],yy[0,iymin:iymax+1])
        
        # Although inefficient I am braking this up into 4 different loops because
        # I am concerned about memory usage for a long range lidar
        
        idx = np.identity(len(x))*np.arange(1,len(x)+1)
        
        for j in range(2):
            u = interp1d(q[j],u,axis=j,bounds_error=False)(qi[j])
            idx = np.delete(idx,np.where(np.isnan(u))[j],axis=j)
            u = np.delete(u,np.where(np.isnan(u))[j],axis=j)
        
        foo = np.where(idx != 0)
              
        u = u[foo[0],foo[1],:]
        
        for j in range(2):
            v = interp1d(q[j],v,axis=j,bounds_error=False)(qi[j])
            v = np.delete(v,np.where(np.isnan(v))[j],axis=j)
        
        v = v[foo[0],foo[1],:]
        
        for j in range(2):
            w = interp1d(q[j],w,axis=j,bounds_error=False)(qi[j])
            w = np.delete(w,np.where(np.isnan(w))[j],axis=j)
        
        w = w[foo[0],foo[1],:]
        
        
        for j in range(2):
            zzz = interp1d(q[j],zzz,axis=j,bounds_error=False)(qi[j])
            zzz = np.delete(zzz,np.where(np.isnan(zzz))[j],axis=j)
        
        zzz = zzz[foo[0],foo[1],:]
        
        for j in range(2):
            ground = interp1d(q[j],ground,axis=j,bounds_error=False)(qi[j])
            ground = np.delete(ground,np.where(np.isnan(ground)[j]),axis=j)
        
        ground = ground[foo[0],foo[1]]
        
        idx = (idx[foo] - 1).astype(int)
    
        vr = []
        ground_free = True
        for i in range(len(x)):
            foo = np.where(i == idx)[0]
            
            if len(foo) == 0:
                vr.append(np.nan)
            else:
                if ground[foo[0]] > z[i]:
                    ground_free = False
                    
                if ground_free:
                    temp_u = np.interp(z[i],zzz[foo[0]],u[foo[0]],left=np.nan,right=np.nan)
                    temp_v = np.interp(z[i],zzz[foo[0]],v[foo[0]],left=np.nan,right=np.nan)
                    temp_w = np.interp(z[i],zzz[foo[0]],w[foo[0]],left=np.nan,right=np.nan)
            
                    vr.append(((x[i]-lidarx)*temp_u + (y[i]-lidary)*temp_v + (z[i]-lidarz)*temp_w)/np.sqrt((x[i]-lidarx)**2 + (y[i]-lidary)**2 + (z[i]-lidarz)**2))
                else:
                    vr.append(np.nan)
        return np.array(vr) 
    
    ##############################################################################
    # Get NCAR LES data
    ##############################################################################
    
    def get_ncarles_data(x,y,z,lidarx,lidary,lidarz,file,nscl):
        
        #This is the number of bytes before the stuff we want
        con_num = (8*(18 + 2*nscl)) + 4
        
        #First read in the constant file to get the grid dimensions
        f = open(file + '.con','rb')
        f.seek(con_num)
        bts = f.read(64)
        temp = struct.unpack('d'*8,bts)
        f.close()
        
        nx = int(temp[2]/temp[5])
        ny = int(temp[3]/temp[6])
        nz = int(temp[4]/temp[7])
        
        xx, yy = np.meshgrid(np.arange(nx) * temp[5], np.arange(ny) * temp[6])
        
        # Come back to this sometime as this may be unnecessary depending how levels
        # are always defined in NCAR LES.
        zz = ((np.arange(nz) * temp[7]) + temp[0])[:,None,None] * np.ones(xx.shape)[None,:,:]
        
        zz = zz.T
        xx = xx.T
        yy = yy.T
        
        # First check if max x is less than the minimum of the grid or if min x is greater than max x
        # and if so just return a bunch of nans because the beam is never in the domain
        if (np.max(x) < np.min(xx[:,0])) or (np.min(x) > np.max(xx[:,0])):
            return np.ones(len(x))*np.nan
        
        # Do the same thing for y
        if (np.max(y) < np.min(yy[0])) or (np.min(y) > np.max(yy[0])):
            return np.ones(len(x))*np.nan
        
        # Search for the indices for the the points needed to find the lidar ray
        if np.max(x) > np.max(xx[:,0]):
            ixmax = nx-1
            if np.min(x) < np.min(xx[:,0]):
                ixmin = 0
            else:
                ixmin = np.where((np.min(x) > xx[:,0]))[0][-1]
        elif np.min(x) < np.min(xx[:,0]):
            ixmin = 0
            ixmax = np.where((np.max(x) <= xx[:,0]))[0][0]
        else:
            ixmin, ixmax = np.where((np.min(x) > xx[:,0]))[0][-1], np.where((np.max(x) <= xx[:,0]))[0][0]
        
        
        if np.max(y) > np.max(yy[0]):
            iymax = ny-1
            if np.min(y) < np.min(yy[0]):
                iymin = 0
            else:
                iymin = np.where((np.min(y) > yy[0]))[0][-1]
        elif np.min(y) < np.min(yy[0]):
            iymin = 0
            iymax = np.where((np.max(y) <= yy[0]))[0][0]
        else:
            iymin, iymax = np.where((np.min(y) > yy[0]))[0][-1], np.where((np.max(y) <= yy[0]))[0][0]
        
        if np.max(z) > np.max(np.min(np.min(zz,axis=0),axis=0)):
            izmax = zz.shape[0]-1
        else:
            izmax = np.where(np.max(z) < np.min(np.min(zz,axis=0),axis=0))[0][0]
        
        u = np.ones((ixmax-ixmin+1,iymax-iymin+1,izmax+1))*np.nan
        v = np.ones((ixmax-ixmin+1,iymax-iymin+1,izmax+1))*np.nan
        w = np.ones((ixmax-ixmin+1,iymax-iymin+1,izmax+2))*np.nan
        
        
        
        f = np.memmap(file,dtype='float',mode='r')
        u = f[::25].reshape(nx,ny,nz+1,order='F')
        v = f[1::25].reshape(nx,ny,nz+1,order='F')
        w = f[2::25].reshape(nx,ny,nz+1,order='F')
        
        u = u[ixmin:ixmax+1,iymin:iymax+1,1:izmax+2]*1
        v = v[ixmin:ixmax+1,iymin:iymax+1,1:izmax+2]*1
        w = w[ixmin:ixmax+1,iymin:iymax+1,:izmax+2]*1
        
        w = (w[:,:,1:] + w[:,:,:-1])/2
        
        del f
        
        zzz = zz[ixmin:ixmax+1,iymin:iymax+1,:izmax+1]
        
        qi = (x,y)
        q = (xx[ixmin:ixmax+1,0],yy[0,iymin:iymax+1])
                
        idx = np.identity(len(x))*np.arange(1,len(x)+1)
        
        for j in range(2):
            u = interp1d(q[j],u,axis=j,bounds_error=False)(qi[j])
            idx = np.delete(idx,np.where(np.isnan(u))[j],axis=j)
            u = np.delete(u,np.where(np.isnan(u))[j],axis=j)
        
        foo = np.where(idx != 0)
              
        u = u[foo[0],foo[1],:]
        
        for j in range(2):
            v = interp1d(q[j],v,axis=j,bounds_error=False)(qi[j])
            v = np.delete(v,np.where(np.isnan(v))[j],axis=j)
        
        v = v[foo[0],foo[1],:]
        
        for j in range(2):
            w = interp1d(q[j],w,axis=j,bounds_error=False)(qi[j])
            w = np.delete(w,np.where(np.isnan(w))[j],axis=j)
        
        w = w[foo[0],foo[1],:]
        
        
        for j in range(2):
            zzz = interp1d(q[j],zzz,axis=j,bounds_error=False)(qi[j])
            zzz = np.delete(zzz,np.where(np.isnan(zzz))[j],axis=j)
        
        zzz = zzz[foo[0],foo[1],:]
        
        idx = (idx[foo] - 1).astype(int)
    
        vr = []
        for i in range(len(x)):
            foo = np.where(i == idx)[0]
            
            if len(foo) == 0:
                vr.append(np.nan)
            else:
                temp_u = np.interp(z[i],zzz[foo[0]],u[foo[0]],left=np.nan,right=np.nan)
                temp_v = np.interp(z[i],zzz[foo[0]],v[foo[0]],left=np.nan,right=np.nan)
                temp_w = np.interp(z[i],zzz[foo[0]],w[foo[0]],left=np.nan,right=np.nan)
            
                vr.append(((x[i]-lidarx)*temp_u + (y[i]-lidary)*temp_v + (z[i]-lidarz)*temp_w)/np.sqrt((x[i]-lidarx)**2 + (y[i]-lidary)**2 + (z[i]-lidarz)**2))
    
        return np.array(vr)
    
    ###############################################################################
    # Get data from a MicroHH run.
    ###############################################################################
    def Get_model_data(lidar_parameter):
        if lidar_parameter['model_type'] == 4:
            foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '_u-003.nc')[0][0]
            u_file = lidar_parameter['files'][foo]            
            foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '_v-002.nc')[0][0]
            v_file = lidar_parameter['files'][foo]            
            foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '_w-001.nc')[0][0]
            w_file = lidar_parameter['files'][foo]            
            # temp = Read_file.get_MicroHH_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['model_time'], u_file, v_file, w_file)
            model_data = Read_file.Read_MicroHH_data(lidar_parameter['model_time'], u_file,v_file,w_file)
            
        return model_data
    def Model_interpolation(ray, lidar_parameter,model_data):
            if lidar_parameter['model_type'] == 1:
                #try:
                foo = np.flatnonzero(np.core.defchararray.find(np.array(lidar_parameter['files']),lidar_parameter['model_time'].strftime('%Y-%m-%d_%H:%M:%S'))!=-1)[0]
                # temp = Read_file.get_wrf_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['files'][foo],lidar_parameter['clouds'],xx,yy,transform)
                # except:
                #     print('ERROR: No model data was found for ' + str(model_time))
                #     sys.exit()
            
            elif lidar_parameter['model_type'] == 2:
                foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '.' + str(int(lidar_parameter['model_time']/lidar_parameter['model_step'])))[0][0]
                # temp = Read_file.get_fasteddy_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['files'][foo])
                
            elif lidar_parameter['model_type'] == 3:
                if int(lidar_parameter['model_time']/lidar_parameter['model_frequency']) < 10:
                    foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '.00' + str(int(lidar_parameter['model_time']/lidar_parameter['model_frequency'])))[0][0]
                elif int(lidar_parameter['model_time']/lidar_parameter['model_frequency']) < 100:
                    foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '.0'  + str(int(lidar_parameter['model_time']/lidar_parameter['model_frequency'])))[0][0]
                else:
                    foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '.'   + str(int(lidar_parameter['model_time']/lidar_parameter['model_frequency'])))[0][0]
                
                # temp = Read_file.get_ncarles_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['files'][foo],lidar_parameter['nscl'])
            
            elif lidar_parameter['model_type'] == 4:
                temp = Read_file.get_MicroHH_data2(ray,lidar_parameter,model_data)

            elif lidar_parameter['model_type'] == 5:
                print(lidar_parameter['model_time'], lidar_parameter['model_frequency'])
                foo = 0 
                time_step = np.floor(lidar_parameter['model_time']/lidar_parameter['model_frequency'])
            else:
                print('ERROR: Unknown model type specified')
                return [-999]
        
        
            return temp
    def Read_MicroHH_data(model_time, u_file,v_file,w_file):
        model_data={}
        
        fu = Dataset(u_file)        
        model_data['t'] = fu['time'][:]        
        foo = np.where(model_data['t'] == model_time)[0][0]        
        model_data['zz'] = fu['z'][:-1]
        model_data['yy'] = fu['y'][:-1]
        
        fv = Dataset(v_file)        
        model_data['xx']= fv['x'][:-1]        
        fw = Dataset(w_file)
        
        model_data['u'] = fu['u'][foo]
        model_data['v'] = fv['v'][foo]
        model_data['w'] = fw['w'][foo]
        
        
        model_data['xx'], model_data['yy'] = np.meshgrid(model_data['xx'],  model_data['yy'])
        # print(model_data['xx'].shape)
        # sys.exit()
        model_data['zz'] = model_data['zz'][:,None,None] * np.ones(model_data['xx'].shape)[None,:,:]     
#        zz = zz[:,None,None] * np.ones(xx.shape)[None,:,:]

        model_data['zz'] = model_data['zz'].T
        model_data['xx'] = model_data['xx'].T
        model_data['yy'] = model_data['yy'].T
        fu.close()
        fv.close()
        fw.close()
        return model_data
    def get_MicroHH_data2(ray,lidar_parameter,model_data):
        
        
        # and if so just return a bunch of nans because the beam is never in the domain
        if (np.max(ray['x']) < np.min(model_data['xx'][:,0])) or (np.min(ray['x']) > np.max(model_data['xx'][:,0])):
            return np.ones(len(ray['x']))*np.nan
        
        # Do the same thing for y
        if (np.max(ray['y']) < np.min(model_data['yy'][0])) or (np.min(ray['y']) > np.max(model_data['yy'][0])):
            return np.ones(len(ray['x']))*np.nan
        
        if np.max(ray['x']) > np.max(model_data['xx'][:,0]):
            ixmax = len(model_data['xx'])-1
            if np.minray_(ray['x']) < np.min(model_data['xx'][:,0]):
                ixmin = 0
            else:
                ixmin = np.where((np.min(ray['x']) > model_data['xx'][:,0]))[0][-1]
        elif np.min(ray['x']) < np.min(model_data['xx'][:,0]):
            ixmin = 0
            ixmax = np.where((np.max(ray['x']) <= model_data['xx'][:,0]))[0][0]
        else:
            ixmin, ixmax = np.where((np.min(ray['x']) > model_data['xx'][:,0]))[0][-1], np.where((np.max(ray['x']) <= model_data['xx'][:,0]))[0][0]
        
        if np.max(ray['y']) > np.max(model_data['yy'][0]):
            iymax = len(model_data['yy'])-1
            if np.min(ray['y']) < np.min(model_data['yy'][0]):
                iymin = 0
            else:
                iymin = np.where((np.min(ray['y']) > model_data['yy'][0]))[0][-1]
        elif np.min(ray['y']) < np.min(model_data['yy'][0]):
            iymin = 0
            iymax = np.where((np.max(ray['y']) <= model_data['yy'][0]))[0][0]
        else:
            iymin, iymax = np.where((np.min(ray['y']) > model_data['yy'][0]))[0][-1], np.where((np.max(ray['y']) <= model_data['yy'][0]))[0][0]
        
        if np.max(ray['z']) > np.max(np.min(np.min(model_data['zz'],axis=0),axis=0)):
            izmax = model_data['zz'].shape[0]-1
        else:
            izmax = np.where(np.max(ray['z']) < np.min(np.min(model_data['zz'],axis=0),axis=0))[0][0]
       
        u = model_data['u']#[foo] 
        u = (u[:-1,:-1,1:] + u[:-1,:-1,:-1])/2.
        u = u[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        # fu.close()
     
        v = model_data['v']#[foo]
        v = (v[:-1,1:,:-1] + v[:-1,:-1,:-1])/2.
        v = v[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        # fv.close()
      
        w = model_data['w']#[foo]
        w = (w[1:,:-1,:-1] + w[:-1,:-1,:-1])/2.
        w = w[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        # fw.close()
        
        zzz = model_data['zz'][ixmin:ixmax+1,iymin:iymax+1,:izmax+1]
        
        qi = (ray['x'],ray['y'])
        # print(qi[0].shape)
        # print(qi[1])
        # print(xx.shape)
        # print(yy.shape)
        q = (model_data['xx'][ixmin:ixmax+1,0],model_data['yy'][0,iymin:iymax+1])
        # print(q[0].shape)
        # print(q[1].shape)
        # sys.exit()
        # print(q)
        # sys.exit()
        idx = np.identity(len(ray['x']))*np.arange(1,len(ray['x'])+1)
        
        for j in range(2):
            u = interp1d(q[j],u,axis=j,bounds_error=False)(qi[j])
            idx = np.delete(idx,np.where(np.isnan(u))[j],axis=j)
            u = np.delete(u,np.where(np.isnan(u))[j],axis=j)

        foo = np.where(idx != 0)
    
        u = u[foo[0],foo[1],:]
    
        for j in range(2):
            v = interp1d(q[j],v,axis=j,bounds_error=False)(qi[j])
            v = np.delete(v,np.where(np.isnan(v))[j],axis=j)
        
        v = v[foo[0],foo[1],:]
        
        for j in range(2):
            w = interp1d(q[j],w,axis=j,bounds_error=False)(qi[j])
            w = np.delete(w,np.where(np.isnan(w))[j],axis=j)
        
        w = w[foo[0],foo[1],:]
        
        for j in range(2):
            zzz = interp1d(q[j],zzz,axis=j,bounds_error=False)(qi[j])
            zzz = np.delete(zzz,np.where(np.isnan(zzz))[j],axis=j)
        
        zzz = zzz[foo[0],foo[1],:]
        
        idx = (idx[foo] - 1).astype(int)
        # print(foo[0].shape)
        # print(foo[1].shape)

        # print(w.shape)
        # sys.exit()
        vr = []
        for i in range(len(ray['x'])):
            foo = np.where(i == idx)[0]
            # print(i,zzz[foo[0]])
            
            if len(foo) == 0:
                vr.append(np.nan)
            else:
                temp_u = np.interp(ray['z'][i],zzz[foo[0]],u[foo[0]],left=np.nan,right=np.nan)
                temp_v = np.interp(ray['z'][i],zzz[foo[0]],v[foo[0]],left=np.nan,right=np.nan)
                temp_w = np.interp(ray['z'][i],zzz[foo[0]],w[foo[0]],left=np.nan,right=np.nan)
            # lidar_parameter['lidarx'],lidar_parameter['lidary'],lidar_parameter['lidarz']
                vr.append((( ray['x'][i] - lidar_parameter['lidar_x'] ) * temp_u +
                           ( ray['y'][i] - lidar_parameter['lidar_y'] ) * temp_v + 
                           ( ray['z'][i] - lidar_parameter['lidar_z'] ) * temp_w) /
                             np.sqrt(( ray['x'][i] - lidar_parameter['lidar_x'])**2 +
                                     ( ray['y'][i] - lidar_parameter['lidar_y'])**2 +
                                     ( ray['z'][i] - lidar_parameter['lidar_z'])**2))
                # vr.append(((x[i]-lidarx)*temp_u + (y[i]-lidary)*temp_v + (z[i]-lidarz)*temp_w)/
                          # np.sqrt((x[i]-lidarx)**2 + (y[i]-lidary)**2 + (z[i]-lidarz)**2))
        # sys.exit()
        return np.array(vr)
        

    def get_MicroHH_data(ray_x,ray_y,ray_z,lidarx,lidary,lidarz,model_time, u_file,v_file,w_file):
        fu = Dataset(u_file)
        
        t = fu['time'][:]
        
        foo = np.where(t == model_time)[0][0]
        
        zz = fu['z'][:-1]
        yy = fu['y'][:-1]
        
        fv = Dataset(v_file)
        
        xx = fv['x'][:-1]

        # print(qi[1])
        
        
        fw = Dataset(w_file)
        
        xx, yy = np.meshgrid(xx, yy)
        # print(xx.shape)
        # print(yy.shape)
        print(xx.shape)
        sys.exit()
        zz = zz[:,None,None] * np.ones(xx.shape)[None,:,:]
        
        zz = zz.T
        xx = xx.T
        yy = yy.T
        
        # First check if max x is less than the minimum of the grid or if min x is greater than max x
        # and if so just return a bunch of nans because the beam is never in the domain
        if (np.max(ray_x) < np.min(xx[:,0])) or (np.min(ray_x) > np.max(xx[:,0])):
            return np.ones(len(ray_x))*np.nan
        
        # Do the same thing for y
        if (np.max(ray_y) < np.min(yy[0])) or (np.min(ray_y) > np.max(yy[0])):
            return np.ones(len(ray_x))*np.nan
        
        if np.max(ray_x) > np.max(xx[:,0]):
            ixmax = len(xx)-1
            if np.minray_(ray_x) < np.min(xx[:,0]):
                ixmin = 0
            else:
                ixmin = np.where((np.min(ray_x) > xx[:,0]))[0][-1]
        elif np.min(ray_x) < np.min(xx[:,0]):
            ixmin = 0
            ixmax = np.where((np.max(ray_x) <= xx[:,0]))[0][0]
        else:
            ixmin, ixmax = np.where((np.min(ray_x) > xx[:,0]))[0][-1], np.where((np.max(ray_x) <= xx[:,0]))[0][0]
        
        if np.max(ray_y) > np.max(yy[0]):
            iymax = len(yy)-1
            if np.min(ray_y) < np.min(yy[0]):
                iymin = 0
            else:
                iymin = np.where((np.min(ray_y) > yy[0]))[0][-1]
        elif np.min(ray_y) < np.min(yy[0]):
            iymin = 0
            iymax = np.where((np.max(ray_y) <= yy[0]))[0][0]
        else:
            iymin, iymax = np.where((np.min(ray_y) > yy[0]))[0][-1], np.where((np.max(ray_y) <= yy[0]))[0][0]
        
        if np.max(ray_z) > np.max(np.min(np.min(zz,axis=0),axis=0)):
            izmax = zz.shape[0]-1
        else:
            izmax = np.where(np.max(ray_z) < np.min(np.min(zz,axis=0),axis=0))[0][0]
       
        u = fu['u'][foo] 
        u = (u[:-1,:-1,1:] + u[:-1,:-1,:-1])/2.
        u = u[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        fu.close()
     
        v = fv['v'][foo]
        v = (v[:-1,1:,:-1] + v[:-1,:-1,:-1])/2.
        v = v[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        fv.close()
      
        w = fw['w'][foo]
        w = (w[1:,:-1,:-1] + w[:-1,:-1,:-1])/2.
        w = w[:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        fw.close()
        
        zzz = zz[ixmin:ixmax+1,iymin:iymax+1,:izmax+1]
        
        qi = (ray_x,ray_y)
        # print(qi[0].shape)
        # print(qi[1])
        # print(xx.shape)
        # print(yy.shape)
        q = (xx[ixmin:ixmax+1,0],yy[0,iymin:iymax+1])
        # print(q[0].shape)
        # print(q[1].shape)
        # sys.exit()
        # print(q)
        # sys.exit()
        idx = np.identity(len(ray_x))*np.arange(1,len(ray_x)+1)
        
        for j in range(2):
            u = interp1d(q[j],u,axis=j,bounds_error=False)(qi[j])
            idx = np.delete(idx,np.where(np.isnan(u))[j],axis=j)
            u = np.delete(u,np.where(np.isnan(u))[j],axis=j)

        foo = np.where(idx != 0)
    
        u = u[foo[0],foo[1],:]
    
        for j in range(2):
            v = interp1d(q[j],v,axis=j,bounds_error=False)(qi[j])
            v = np.delete(v,np.where(np.isnan(v))[j],axis=j)
        
        v = v[foo[0],foo[1],:]
        
        for j in range(2):
            w = interp1d(q[j],w,axis=j,bounds_error=False)(qi[j])
            w = np.delete(w,np.where(np.isnan(w))[j],axis=j)
        
        w = w[foo[0],foo[1],:]
        
        for j in range(2):
            zzz = interp1d(q[j],zzz,axis=j,bounds_error=False)(qi[j])
            zzz = np.delete(zzz,np.where(np.isnan(zzz))[j],axis=j)
        
        zzz = zzz[foo[0],foo[1],:]
        
        idx = (idx[foo] - 1).astype(int)
        # print(foo[0].shape)
        # print(foo[1].shape)

        # print(w.shape)
        # sys.exit()
        vr = []
        for i in range(len(ray_x)):
            foo = np.where(i == idx)[0]
            # print(i,zzz[foo[0]])
            
            if len(foo) == 0:
                vr.append(np.nan)
            else:
                temp_u = np.interp(ray_z[i],zzz[foo[0]],u[foo[0]],left=np.nan,right=np.nan)
                temp_v = np.interp(ray_z[i],zzz[foo[0]],v[foo[0]],left=np.nan,right=np.nan)
                temp_w = np.interp(ray_z[i],zzz[foo[0]],w[foo[0]],left=np.nan,right=np.nan)
            
                vr.append((( ray_x[i] - lidarx ) * temp_u + ( ray_y[i] - lidary ) * temp_v + ( ray_z[i] - lidarz ) * temp_w) /
                              np.sqrt(( ray_x[i] - lidarx )**2 + ( ray_y[i] - lidary)**2 + ( ray_z[i] - lidarz)**2))
                # vr.append(((x[i]-lidarx)*temp_u + (y[i]-lidary)*temp_v + (z[i]-lidarz)*temp_w)/
                          # np.sqrt((x[i]-lidarx)**2 + (y[i]-lidary)**2 + (z[i]-lidarz)**2))
        # sys.exit()
        return np.array(vr)
        
    ##############################################################################
    # Get the data you need from a CM1 simulation
    ##############################################################################
    
    def get_cm1_data(x,y,z,lidarx,lidary,lidarz,file,time_step,cloud,turb,az=None,el=None):
        
        if (turb == 1) and ((az is None) or (el is None)):
            raise Exception('The azimuth and elevation need to be passed to function if subgrid turbulence is to be used')
            
        f = Dataset(file,'r')
        
        # zz = f['z'][:]*1000.
        zz = f['zh'][:]*1000.
        xx = f['xh'][:]*1000.
        yy = f['yh'][:]*1000.
        
        zz = zz.T
        xx = xx.T
        yy = yy.T
        
        if np.max(x) > np.max(xx):
            ixmax = len(xx)-1
            if np.min(x) < np.min(xx):
                ixmin = 0
            else:
                ixmin = np.where((np.min(x) > xx))[0][-1]
        elif np.min(x) < np.min(xx):
            ixmin = 0
            ixmax = np.where((np.max(x) <= xx))[0][0]
        else:
            ixmin, ixmax = np.where((np.min(x) > xx))[0][-1], np.where((np.max(x) <= xx))[0][0]
        
        
        if np.max(y) > np.max(yy):
            iymax = len(yy)-1
            if np.min(y) < np.min(yy):
                iymin = 0
            else:
                iymin = np.where((np.min(y) > yy))[0][-1]
        elif np.min(y) < np.min(yy):
            iymin = 0
            iymax = np.where((np.max(y) <= yy))[0][0]
        else:
            iymin, iymax = np.where((np.min(y) > yy))[0][-1], np.where((np.max(y) <= yy))[0][0]
        
        if np.max(z) > np.max(zz):
            izmax = len(zz)-1
        else:
            izmax = np.where(np.max(z) < zz)[0][0]
        
        u = f['uinterp'][time_step,:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        v = f['vinterp'][time_step,:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        w = f['winterp'][time_step,:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T
        
        zzz = zz[:izmax+1,None,None] * np.ones((len(yy[iymin:iymax+1]),len(xx[ixmin:ixmax+1])))[None,:,:]
        
        zzz = zzz.T
        
        qi = (x,y)
        q = (xx[ixmin:ixmax+1],yy[iymin:iymax+1])

        # Although inefficient I am braking this up into 4 different loops because
        # I am concerned about memory usage for a long range lidar
        
        idx = np.identity(len(x))*np.arange(1,len(x)+1)
        
        for j in range(2):
            u = interp1d(q[j],u,axis=j,bounds_error=False)(qi[j])
            idx = np.delete(idx,np.where(np.isnan(u))[j],axis=j)
            u = np.delete(u,np.where(np.isnan(u))[j],axis=j)
         
        foo = np.where(idx != 0)
              
        u = u[foo[0],foo[1],:]
        
        for j in range(2):
            v = interp1d(q[j],v,axis=j,bounds_error=False)(qi[j])
            v = np.delete(v,np.where(np.isnan(v))[j],axis=j)
        
        v = v[foo[0],foo[1],:]
        
        for j in range(2):
            w = interp1d(q[j],w,axis=j,bounds_error=False)(qi[j])
            w = np.delete(w,np.where(np.isnan(w))[j],axis=j)
        
        w = w[foo[0],foo[1],:]
        
        for j in range(2):
            zzz = interp1d(q[j],zzz,axis=j,bounds_error=False)(qi[j])
            zzz = np.delete(zzz,np.where(np.isnan(zzz))[j],axis=j)
        
        zzz = zzz[foo[0],foo[1],:]
        
        if cloud == 1:
            # print('cloud')
            qtotal  = (f['qc'][time_step,:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T +
                       f['qi'][time_step,:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T +
                       f['qr'][time_step,:izmax+1,iymin:iymax+1,ixmin:ixmax+1].T)
            
            for j in range(2):
                qtotal = interp1d(q[j],qtotal,axis=j,bounds_error=False)(qi[j])
                qtotal = np.delete(qtotal,np.where(np.isnan(qtotal))[j],axis=j)
            
            qtotal = qtotal[foo[0],foo[1],:]
        
        if turb == 1:
             # Calculate the vr variance before interpolation
             vr_var = (f['uu'][0,ixmin:ixmax+1,iymin:iymax+1,:izmax+1]*
                       (np.cos(np.deg2rad(el))**2)*(np.sin(np.deg2rad(az))**2))
             
             vr_var = vr_var + (f['vv'][0,ixmin:ixmax+1,iymin:iymax+1,:izmax+1]*
                                (np.cos(np.deg2rad(el))**2)*(np.cos(np.deg2rad(az))**2))
             
             
             vr_var = vr_var + (f['ww'][0,ixmin:ixmax+1,iymin:iymax+1,:izmax+1]*
                                (np.sin(np.deg2rad(el))**2))
             
             # Trash is just a temp variable
             trash = 0.5*(f['uv'][0,:,:,1:] + f['uv'][0,:,:,:-1])
             trash = 0.5*(trash[:,1:,:] + trash[:,:-1,:])
             
             vr_var = vr_var + (trash[ixmin:ixmax+1,iymin:iymax+1,:izmax+1]*
                                (np.cos(np.deg2rad(el))**2)*np.cos(np.deg2rad(az))*np.sin(np.deg2rad(az)))
             
             
             trash = 0.5*(f['uw'][0,:,:,1:] + f['uw'][0,:,:,:-1])
             trash = 0.5*(trash[1:,:,:] + trash[:-1,:,:])
             
             vr_var = vr_var + (trash[ixmin:ixmax+1,iymin:iymax+1,:izmax+1]*
                                np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))*np.sin(np.deg2rad(az)))
             
    
             trash = 0.5*(f['vw'][0,:,1:,:] + f['vw'][0,:,:-1,:])
             trash = 0.5*(trash[1:,:,:] + trash[:-1,:,:])
             
             vr_var = vr_var + (trash[ixmin:ixmax+1,iymin:iymax+1,:izmax+1]*
                                np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))*np.cos(np.deg2rad(az)))
             
             trash = 0
             
             for j in range(2):
                 vr_var = interp1d(q[j],vr_var,axis=j,bounds_error=False)(qi[j])
                 vr_var = np.delete(vr_var,np.where(np.isnan(vr_var))[j],axis=j)
             
             vr_var = vr_var[foo[0],foo[1],:]
             vr_var[vr_var < 0] = 0
             
        f.close()
        
        idx = (idx[foo] - 1).astype(int)
    
        vr = []
        if turb == 1:
            rng = np.random.default_rng()
            
        cloud_free = True
        for i in range(len(x)):
            foo = np.where(i == idx)[0]
            
            if len(foo) == 0:
                vr.append(np.nan)
            else:
                if cloud == 1:
                    temp_q = np.interp(z[i],zzz[foo[0]],qtotal[foo[0]],left=np.nan,right=np.nan)
                    if temp_q > 0.00001:
                        print('Cloud at ' + str(z[i]) + ' qvalue: ' + str(temp_q))
                        cloud_free = False
                    
                if cloud_free:
                    temp_u = np.interp(z[i],zzz[foo[0]],u[foo[0]],left=np.nan,right=np.nan)
                    temp_v = np.interp(z[i],zzz[foo[0]],v[foo[0]],left=np.nan,right=np.nan)
                    temp_w = np.interp(z[i],zzz[foo[0]],w[foo[0]],left=np.nan,right=np.nan)
            
                    vr.append(((x[i]-lidarx)*temp_u + (y[i]-lidary)*temp_v + (z[i]-lidarz)*temp_w)/np.sqrt((x[i]-lidarx)**2 + (y[i]-lidary)**2 + (z[i]-lidarz)**2))
                    
                    if turb == 1:
                        temp_vr_var = np.interp(z[i],zzz[foo[0]],vr_var[foo[0]],left=np.nan,right=np.nan)
                        
                        vr[-1] = vr[-1] + rng.normal(0,np.sqrt(temp_vr_var))
                        
                else:
                    vr.append(np.nan)
    
        return np.array(vr)
#-------------------------------------------------------------------------------
class Time:
    def __init__(self):     
        pass
    def get_scan_timing(scans, start_time, end_time, model_time, cced, repeats, stare_length,
                        scan_speeds,ray_time,instantaneous_scan,use_calendar):
        
        scan_type = []
        scan_schedule = {'scan':[],'start':[],'end':[],'start_index':[],'end_index':[]}
        # This first section is if the user wants instantaneous lidar scans
        # This is an easy case as all elevations and azimuths will be collected at
        # each model time
        
        if instantaneous_scan == 1:
            if stare_length > 0:
                insta_scan = np.array([[0,90]])
                scan_type.append(0)
                
                for i in range(len(scans)):
                    insta_scan = np.append(insta_scan,scans[i],axis=0)
                    scan_type.extend(scans[i].shape[0]*[i+1])
            
            else:
                for i in range(len(scans)):
                    if i == 0:
                        insta_scan = np.copy(scans[i])
                        scan_type.extend(scans[i].shape[0]*[i+1])
                    else:
                        insta_scan = np.append(insta_scan,scans[i],axis=0)
                        scan_type.extend(scans[i].shape[0]*[i+1])
            
            az_el_coords = [insta_scan]*len(model_time)
            scan_key = [scan_type]*len(model_time)
            model_time_key = np.arange(0,len(model_time))
            if use_calendar == 1:
                lidar_time = np.array([(x - datetime(1970,1,1)).total_seconds() for x in model_time])
            else:
                lidar_time = np.array(model_time)
        
        # This is the more complicated section. We need to account for the scanning
        # of the lidar when determining the timing of each azimuth, elevation pair.
        else:
            
            if use_calendar == 1:
                temp_time = (start_time - datetime(1970,1,1)).total_seconds()
                end = (end_time - datetime(1970,1,1)).total_seconds()
                temp_model_time = np.array([(x - datetime(1970,1,1)).total_seconds() for x in model_time])
            else:
                temp_time = start_time
                end = end_time
                temp_model_time = np.copy(model_time)
            
            if stare_length > 0:
                new_cced = np.insert(cced,0,1)
            else:
                new_cced = np.insert(cced,0,0)
            
            # If the user has stare_length = 0, but there are no cc'ed scans
            # then the we need to have filler vertical stares. Note: this will
            # increase computing time since we have to assume 1 second stares. We
            # will warn the user and recommend changing the stare_length
            if len(np.where(new_cced==1)[0]) == 0:
                new_cced[0] = 1
                stare_length = 1
                print("WARNING: No cc'ed scans and stare_length = 0. Setting stare_length = 1.")
                print("         This will increase computing time. Recommend setting stare_length")
                print("         to the longest possible time between repeated scans.")
                
            timer = np.zeros(len(new_cced))
            temp_status  = np.ones(len(new_cced))
            new_repeats = np.insert(repeats,0,0)
            if stare_length == 0:
                temp_status[0] = 0
    
    
            # This loop (ugh... hopefully I come up with a better way in the future)
            # finds the time of each az/el pair. It accounts for cc'ed scans and 
            # repeats through status arrays. If a scan is supposed to repeat every
            # ten minutes but another scan is running then that scan will run
            # after the other scan is finished. CC'ed scan count as one entire scan
            # and lower numbered scans are given priority if both are up for repeat
            # at the same time. This means that a scan that is to be repeated every
            # 10 minutes will most likely only repeat a little after 10 minutes due 
            # to other scans finishing. This is how the scans scheduling works with
            # Halo lidars
            
            
            first = True
            lidar_time = []
            az_el_coords = []
            scan_key = []
            no_cced_index = np.where((new_cced == 0))[0]
            while temp_time <= end:
                for i in range(len(new_cced)):
                  if temp_status[i] == 1:
                    if first:
                        foo = np.where(temp_status == 1)[0][0]
                        if foo == 0:
                            start_coords = np.array([0,90])
                        else:
                            start_coords = scans[foo-1][0,:]
                    
                    if i == 0:
                        first_az_dist = 180.0 - np.abs(np.abs(0-start_coords[0])-180.0)
                        first_el_dist = np.abs(90-start_coords[1])
                        if first:
                            scan_time = np.ones(int(stare_length))*ray_time
                            scan_time[0] += ((first_az_dist/scan_speeds[0,0]) + (first_el_dist/scan_speeds[0,1]) - ray_time)
                            scan_schedule['start_index'].append(0)
                            first = False
                        else:
                            scan_time = np.ones(int(stare_length))*ray_time
                            scan_time[0] += ((first_az_dist/scan_speeds[0,0]) + (first_el_dist/scan_speeds[0,1]))
                            scan_schedule['start_index'].append(len(az_el_coords))
                        az_el_coords.extend([np.array([0,90])]*int(stare_length))
                        scan_key.extend([0]*int(stare_length))
                        start_coords = np.array([0,90])
                        scan_schedule['scan'].append('Stare')
                        
                    
                    else:
                        first_az_dist = 180.0 - np.abs(np.abs(scans[i-1][0,0]-start_coords[0])-180.0)
                        first_el_dist = np.abs(scans[i-1][0,1]-start_coords[1])
                        az_dist = 180.0 - np.abs(np.abs(scans[i-1][1:,0]-scans[i-1][:-1,0]) - 180.0)
                        el_dist = np.abs(scans[i-1][1:,1]-scans[i-1][:-1,1])
                        scan_time = np.ones(len(az_dist)+1)*np.nan
                        if first:
                            scan_time[0] = (first_az_dist/scan_speeds[0,0]) + (first_el_dist/scan_speeds[0,1])
                            first = False
                            scan_schedule['start_index'].append(0)
                        else:
                            scan_time[0] = (first_az_dist/scan_speeds[0,0]) + (first_el_dist/scan_speeds[0,1]) + ray_time  
                            scan_schedule['start_index'].append(len(az_el_coords))
                        
                        if (scan_speeds[i,0] == -1) | (scan_speeds[i,1] == -1):
                            scan_time[1:] = ray_time
                        else:
                            scan_time[1:] = (az_dist/scan_speeds[i,0]) + (el_dist/scan_speeds[i,1]) + ray_time
                        
                        az_el_coords.extend(scans[i-1])
                        scan_key.extend([i]*scans[i-1].shape[0])
                        start_coords = np.copy(scans[i-1][-1])
                        scan_schedule['scan'].append('Scan '+str(i))
                        
                    
                    lidar_time.extend(temp_time+np.cumsum(scan_time))
                    timer[no_cced_index] += np.cumsum(scan_time)[-1]
                    
                    if new_cced[i] == 0:
                        temp_status[i] = 0
                        timer[i] = np.cumsum(scan_time)[-1]
                    
                    foo = np.where(((timer[1:]-new_repeats[1:]) >= 0) & (new_cced[1:] == 0))[0]
                    temp_status[foo+1] = 1
                    scan_schedule['start'].append(temp_time)
                    scan_schedule['end'].append(lidar_time[-1])
                    scan_schedule['end_index'].append(len(az_el_coords)-1)
                    temp_time = lidar_time[-1]
            
            model_time_key = [np.argmin(np.abs(x-temp_model_time)) for x in np.array(lidar_time)]
           
        return az_el_coords, scan_key, model_time_key, lidar_time, scan_schedule

class Signal:
    # a = 1
    # b = 2
    def __init__(self):     
        pass

    ###############################################################################
    # This function uses a signal climatology file to make more "realistic" signal
    # loss for the simulated lidar data.
    ###############################################################################
    def Range_Weighting_function(pulse_width,gate_width):
        c =3e8
        # pulse_width = namelist['pulse_width'] 
        # gate_width = namelist['gate_width']
        r_high = np.arange(1,5001)
        # from scipy.special import erf

        rwf = (1/(c*gate_width*1e-9))*\
              (erf((4*np.sqrt(np.log(2))*(r_high -3000)/(c*pulse_width*1e-9)) + np.sqrt(np.log(2))*gate_width/pulse_width)-\
               erf((4*np.sqrt(np.log(2))*(r_high -3000)/(c*pulse_width*1e-9)) - np.sqrt(np.log(2))*gate_width/pulse_width))
        return rwf
            
        
        
    def signal_sim(ray, file, r, nyquist_velocity, points_per_gate, num_pulse):
        
        vr = np.copy(ray)
        f = Dataset(file,'r')
        
        climo_snr = f.variables['snr'][:]
        good = f.variables['fraction_good'][:]
        climo_r = f.variables['ranges'][:]
        
        # Interpolate the climo signal to the ranges of the simulated lidar data
        
        climo_snr = np.interp(r, climo_r, climo_snr)
        good = np.interp(r, climo_r, good)
        # Generate a random float between 0 and 1. If the number is greater than
        # "good" then replace that value with one from a beta distribution
        
        
        rng = np.random.default_rng()
        random = rng.random(len(r))
        beta = rng.beta(0.5,0.5,len(r))
        beta = (beta*(2*nyquist_velocity)) - nyquist_velocity
        
        # Replace any nans in the dataset with junk values
        foo = np.where(np.isnan(vr))
        vr[foo] = beta[foo]
        
        # Now do signal climo
        foo = np.where(random > good)
        vr[foo] = beta[foo]
        
        # Calculate the observation error variance for each range gate based on 
        # Eq. 7 from O'Conner (2010)
        alpha = ((10**(climo_snr/10.)) * 2*nyquist_velocity)/(2*np.sqrt(2*np.pi))
        obs_var = (((4*np.sqrt(8))/(alpha*(10**(climo_snr/10.))*points_per_gate*num_pulse))*
                   ((1 + (alpha/np.sqrt(2*np.pi)))**2))
        
        # Now add Gaussian perturbations to the Vr depending on the SNR values for
        # "good" points
        foo = np.where(random <= good)
        normal = rng.normal(0,np.sqrt(obs_var))
        vr[foo] += normal[foo]
        
        # Now check to make sure none of the data is past the nyquist velocity
        vr = np.where(vr >= -1*nyquist_velocity, vr, -1*nyquist_velocity)
        vr = np.where(vr <= nyquist_velocity, vr, nyquist_velocity)
        
        return vr
    ##############################################################################
    # This function samples model data to the specified range gates assuming a 
    # gaussian pusle power
    ##############################################################################

    def gaussian_pulse(vr,lidar_parameter,cut_off,r_sample, key, elev, namelist):
        
        
        c = 3e8
        r_high = np.arange(1,(lidar_parameter['maximum_range']*1000)+1)
        
        if ((key == 1) & (namelist['dbs1'] == 1)):
            r = np.arange(namelist['dbs_start_height'],namelist['dbs_end_height']
                                   +1,namelist['dbs_spacing'])/np.cos(90-np.deg2rad(elev))
            
        elif ((key == 2) & (namelist['dbs2'] == 1)):
             r = np.arange(namelist['dbs_start_height'],namelist['dbs_end_height']
                                   +1,namelist['dbs_spacing'])/np.cos(90-np.deg2rad(elev))
        
        elif ((key == 3) & (namelist['dbs3'] == 1)):
             r = np.arange(namelist['dbs_start_height'],namelist['dbs_end_height']
                                   +1,namelist['dbs_spacing'])/np.cos(90-np.deg2rad(elev))
        
        elif ((key == 4) & (namelist['dbs4'] == 1)):
             r = np.arange(namelist['dbs_start_height'],namelist['dbs_end_height']
                                   +1,namelist['dbs_spacing'])/np.cos(90-np.deg2rad(elev))
        else:
            r = np.arange(3e8 * lidar_parameter['gate_width'] * 1e-9 / 4.0 ,lidar_parameter['maximum_range'] * 1000 + 1, 3e8 * lidar_parameter['gate_width'] * 1e-9 / 2.0)
        
        rwf = (1/(c*lidar_parameter['gate_width'] * 1e-9))*(erf((4 * np.sqrt(np.log(2))*(r_high[None,:] - r[:,None])/(c*lidar_parameter['pulse_width']*1e-9)) + np.sqrt(np.log(2))*lidar_parameter['gate_width']/lidar_parameter['pulse_width'])
                                                           -erf((4 * np.sqrt(np.log(2))*(r_high[None,:] - r[:,None])/(c*lidar_parameter['pulse_width']*1e-9))  -np.sqrt(np.log(2))*lidar_parameter['gate_width']/lidar_parameter['pulse_width']))
        
        # vr is the vr interpolated by the model dimsnsion
        vr_interp = np.interp(r_high,r_sample,vr,left=np.nan,right=np.nan) # all vr intepolated again along the beam
        # print(vr_interp.shape)
        # sys.exit()
        foo = np.where(~np.isnan(vr_interp))[0]
        
        if len(foo) > 0:
            cut_min = foo[0]+1 + cut_off
            cut_max = foo[-1]+1 - cut_off
        else:
            cut_min = cut_off
            cut_max = lidar_parameter['maximum_range'] - cut_off
        
        vr_interp[np.isnan(vr_interp)] = 0.0
        
        
        integrand = vr_interp * rwf # VR * RWF
        
        
        vr_lidar = np.trapz(integrand,axis=1)  # summation along the RWF
        # print('rwf'+str(rwf.shape))
        # print('vr'+str(vr.shape))
        # print('vr_interp'+str(vr_interp.shape))
        # print('vr_lidar '+str(vr_lidar .shape))
        # print('rwf'+vr_interp.size)

        # sys.exit()
       
        foo1 = np.where(r > cut_min)[0]
        foo2 = np.where(r < cut_max)[0]
        
        
        if ((len(foo1) > 0) & (len(foo2) > 0)):
            
            vr_lidar[:foo1[0]] = np.nan
            vr_lidar[foo2[-1]+1:] = np.nan
        else:
            vr_lidar[:] = np.nan
        
        # Filter out the gates with velocity higher than the nyquist velocity
        
        foo = np.where(np.abs(vr_lidar) > lidar_parameter['nyquist_velocity'])[0]
        vr_lidar[foo] = np.nan
        
        # SW calculation---------------------------------------------------------------------------------------------
        
        
        
        
        
        return vr_lidar,vr_lidar
    
    
    ##############################################################################
    # This is the heart of LidarSim. In this function, the simulated lidar observations
    # are obtained. This function calls the read functions for different model types
    # and only reads into memory the data points needed to simulate the rays for that
    # model time.
    ##############################################################################
    
    # def sim_observations(lidar_x, lidar_y, lidar_z, pulse_width, gate_width, sample_resolution, maximum_range,
    #                      nyquist_velocity, coords, model_type, model_time, model_step, files, instantaneous_scan,
    #                      prefix, model_frequency, nscl, clouds, scan_key,
    #                      sim_signal, signal_file, namelist, xx = None, yy = None, transform = None):
    def sim_observations(lidar_parameter, namelist, xx = None, yy = None, transform = None):
        
        if 3e8 * lidar_parameter['gate_width']* 1e-9 / 2.0 < lidar_parameter['sample_resolution']:
            r = np.arange(1,lidar_parameter['maximum_range'] * 1000,3e8 * lidar_parameter['gate_width'] * 1e-9 /2.0)
        else:
            r = np.arange(lidar_parameter['sample_resolution'],lidar_parameter['maximum_range'] * 1000 + lidar_parameter['sample_resolution'],lidar_parameter['sample_resolution'])
        # print(r.shape)
        # Find the maximum and minimum range needed for full gate sample
        # c =3e8
        # r_high = np.arange(1,5001)
        # rwf = (1/(c*gate_width*1e-9))*\
        #       (erf((4*np.sqrt(np.log(2))*(r_high -3000)/(c*pulse_width*1e-9)) + np.sqrt(np.log(2))*gate_width/pulse_width)-\
        #        erf((4*np.sqrt(np.log(2))*(r_high -3000)/(c*pulse_width*1e-9)) - np.sqrt(np.log(2))*gate_width/pulse_width))
        lidar_parameter['rwf'] =  Signal.Range_Weighting_function(lidar_parameter['pulse_width'],lidar_parameter['gate_width'])        
                 
                  
        foo = np.where(lidar_parameter['rwf'] > 0.001)[0]
        #cut_off = int(len(foo)/2)
        cut_off = 0
        # lidar_parameter['
        
        if lidar_parameter['instantaneous_scan'] == 0:
            num = len(lidar_parameter['coords'])
        else:
            num = len(lidar_parameter['coords'][0])
            
        sim_obs_vr = []   
        sim_obs_sw = []    
        # Read the data------------------------------------------------------------
        model_data = Read_file.Get_model_data(lidar_parameter)  # Read the model data

        for j in range(num): # Ray
            print('Ray= '+ str(j))
            if lidar_parameter['instantaneous_scan'] == 0:
                x =  r * np.cos(np.radians(lidar_parameter['coords'][j][1])) * np.sin(np.radians(lidar_parameter['coords'][j][0])) + lidar_parameter['lidar_x']
                y =  r * np.cos(np.radians(lidar_parameter['coords'][j][1])) * np.cos(np.radians(lidar_parameter['coords'][j][0])) + lidar_parameter['lidar_y']
                z =  r * np.sin(np.radians(lidar_parameter['coords'][j][1])) + lidar_parameter['lidar_z']
                key = lidar_parameter['scan_key'][j]
                elev = lidar_parameter['coords'][j][1]
                # azim = lidar_parameter['coords'][j][0]
                
            else:  

                if lidar_parameter['scan_type'] == 0:    #    # 0-Stared-stop mode, 1-Continuous mode(CSM)        
                    x =  r * np.cos(np.radians(lidar_parameter['coords'][0][j,1])) * np.sin(np.radians(lidar_parameter['coords'][0][j,0])) + lidar_parameter['lidar_x']
                    y =  r * np.cos(np.radians(lidar_parameter['coords'][0][j,1])) * np.cos(np.radians(lidar_parameter['coords'][0][j,0])) + lidar_parameter['lidar_y']
                    z =  r * np.sin(np.radians(lidar_parameter['coords'][0][j,1])) + lidar_parameter['lidar_z']
                    key = lidar_parameter['scan_key'][0][j]
                    elev = lidar_parameter['coords'][0][j,1]
                    # azim = lidar_parameter['coords'][0][j,0]
                    ray={'x':x,
                         'y':y,
                         'z':z                     
                          }
                    temp=Read_file.Model_interpolation(ray,lidar_parameter,model_data)
                    temp_vr,temp_sw  = Signal.gaussian_pulse(temp,lidar_parameter,cut_off,r,key,elev,namelist)
                    if lidar_parameter['sim_signal'] == 1:
                        temp_vr,temp_sw = Signal.signal_sim(temp, lidar_parameter['signal_file'], np.arange(3e8 * lidar_parameter['gate_width'] * 1e-9 / 4.0,
                                                 lidar_parameter['maximum_range'] * 1000 + 1, 3e8 * lidar_parameter['gate_width'] * 1e-9 / 2.0),
                                                 lidar_parameter['nyquist_velocity'], namelist['points_per_gate'], namelist['num_pulses'])
                        
                    
                    temp_vr,temp_sw  = Signal.gaussian_pulse(temp,lidar_parameter,cut_off,r,key,elev,namelist)
                elif lidar_parameter['scan_type'] == 1: 
                    pulse_vr = []   
                    pulse_sw = []
                    if lidar_parameter['scan_type_direction'] == 'PPI':
                        az_resolution = abs(lidar_parameter['coords'][0][0,0]-lidar_parameter['coords'][0][0,1])
                        az_pulses = np.linspace(lidar_parameter['coords'][0][j,0]-az_resolution/2,\
                                                lidar_parameter['coords'][0][j,0]+az_resolution/2,
                                                lidar_parameter['psudeo_pulses'])
                        lidar_parameter['coords_peso_pulses'] = np.vstack((az_pulses,np.zeros(az_pulses.size)+lidar_parameter['coords'][0][0,1])).T
                    elif lidar_parameter['scan_type_direction'] == 'RHI':
                        az_resolution = abs(lidar_parameter['coords'][0][0,1]-lidar_parameter['coords'][0][1,1])
                        az_pulses = np.linspace(lidar_parameter['coords'][0][0,j]-az_resolution/2,\
                                                lidar_parameter['coords'][0][0,j]+az_resolution/2,
                                                lidar_parameter['psudeo_pulses'])
                        lidar_parameter['coords_peso_pulses'] = np.vstack((np.zeros(az_pulses.size)+lidar_parameter['coords'][0][0,0],az_pulses)).T

                    for k in range(3): # Different pulses at ray
                        x =  r * np.cos(np.radians(lidar_parameter['coords_peso_pulses'][k,1])) * np.sin(np.radians(lidar_parameter['coords_peso_pulses'][k,0])) + lidar_parameter['lidar_x']
                        y =  r * np.cos(np.radians(lidar_parameter['coords_peso_pulses'][k,1])) * np.cos(np.radians(lidar_parameter['coords_peso_pulses'][k,0])) + lidar_parameter['lidar_y']
                        z =  r * np.sin(np.radians(lidar_parameter['coords_peso_pulses'][k,1])) + lidar_parameter['lidar_z']         
                        
                        key = lidar_parameter['scan_key'][0][j]    
                        elev = lidar_parameter['coords'][0][j,1]
                        ray={'x':x,
                              'y':y,
                              'z':z                     
                              }
                        temp=Read_file.Model_interpolation(ray, lidar_parameter,model_data)
                        if lidar_parameter['sim_signal'] == 1:
                            temp_vr,temp_sw = Signal.signal_sim(temp, lidar_parameter['signal_file'], np.arange(3e8 * lidar_parameter['gate_width'] * 1e-9 / 4.0,
                                                     lidar_parameter['maximum_range'] * 1000 + 1, 3e8 * lidar_parameter['gate_width'] * 1e-9 / 2.0),
                                                     lidar_parameter['nyquist_velocity'], namelist['points_per_gate'], namelist['num_pulses'])
                            
                        
                        temp_pulse_vr,temp_pulse_sw  = Signal.gaussian_pulse(temp,lidar_parameter,cut_off,r,key,elev,namelist)
                        pulse_vr=np.append(pulse_vr,temp_pulse_vr)
                        pulse_sw=np.append(pulse_sw,temp_pulse_sw) 
                        print(lidar_parameter['coords_peso_pulses'][k,0],lidar_parameter['coords_peso_pulses'][k,1])

                    temp_vr  = np.mean(np.reshape(pulse_vr,(k+1,temp_pulse_vr.size)),axis=0)
                    temp_sw  = np.mean(np.reshape(pulse_sw,(k+1,temp_pulse_sw.size)),axis=0)
                    sys.exit()
            sim_obs_vr.append(temp_vr)
            sim_obs_sw.append(temp_sw)
        return sim_obs_vr,sim_obs_sw,temp
    
                    
            
            # if lidar_parameter['model_type'] == 1:
            #     #try:
            #     foo = np.flatnonzero(np.core.defchararray.find(np.array(lidar_parameter['files']),lidar_parameter['model_time'].strftime('%Y-%m-%d_%H:%M:%S'))!=-1)[0]
            #     temp = Read_file.get_wrf_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['files'][foo],lidar_parameter['clouds'],xx,yy,transform)
            #     #except:
            #     #    print('ERROR: No model data was found for ' + str(model_time))
            #     #    sys.exit()
            
            # elif lidar_parameter['model_type'] == 2:
            #     foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '.' + str(int(lidar_parameter['model_time']/lidar_parameter['model_step'])))[0][0]
            #     temp = Read_file.get_fasteddy_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['files'][foo])
                
            # elif lidar_parameter['model_type'] == 3:
            #     if int(lidar_parameter['model_time']/lidar_parameter['model_frequency']) < 10:
            #         foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '.00' + str(int(lidar_parameter['model_time']/lidar_parameter['model_frequency'])))[0][0]
            #     elif int(lidar_parameter['model_time']/lidar_parameter['model_frequency']) < 100:
            #         foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '.0'  + str(int(lidar_parameter['model_time']/lidar_parameter['model_frequency'])))[0][0]
            #     else:
            #         foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '.'   + str(int(lidar_parameter['model_time']/lidar_parameter['model_frequency'])))[0][0]
                
            #     temp = Read_file.get_ncarles_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['files'][foo],lidar_parameter['nscl'])
            
            # elif lidar_parameter['model_type'] == 4:
            #     # foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '_u-003.nc')[0][0]
            #     # # foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + 'u.nc')[0][0]

            #     # u_file = lidar_parameter['files'][foo]
                
            #     # foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '_v-002.nc')[0][0]
            #     # # foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + 'v.nc')[0][0]

            #     # v_file = lidar_parameter['files'][foo]
                
            #     # foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + '_w-001.nc')[0][0]
            #     # # foo = np.where(np.array(lidar_parameter['files']) == lidar_parameter['prefix'] + 'w.nc')[0][0]

            #     # w_file = lidar_parameter['files'][foo]
                
            #     # temp = Read_file.get_MicroHH_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['model_time'], u_file, v_file, w_file)
            #     ray={'x':x,
            #           'y':y,
            #           'z':z                     
            #           }
            #     temp = Read_file.get_MicroHH_data2(ray,lidar_parameter,model_data)

                
                
                
            #     # print(temp.shape)
            #     # sys.exit()
            # elif lidar_parameter['model_type'] == 5:
            #     print(lidar_parameter['model_time'], lidar_parameter['model_frequency'])
            #     # sys.exit()
            #     # Come back to this as it can be done better
            #     # if int(model_time/model_frequency) < 10:
            #     #     print(np.where(np.array(files) == prefix + '_00000' + str(int(model_time/model_frequency))+'.nc'))
            #     #     foo = np.where(np.array(files) == prefix + '_00000' + str(int(model_time/model_frequency))+'.nc')[0][0]
            #     # elif int(model_time/model_frequency) < 100:
            #     #     foo = np.where(np.array(files) == prefix + '_0000' + str(int(model_time/model_frequency))+'.nc')[0][0]
            #     # elif int(model_time/model_frequency) < 1000:
            #     #     foo = np.where(np.array(files) == prefix + '_000' + str(int(model_time/model_frequency))+'.nc')[0][0]
            #     # elif int(model_time/model_frequency) < 10000:
            #     #     foo = np.where(np.array(files) == prefix + '_00' + str(int(model_time/model_frequency))+'.nc')[0][0]
            #     # elif int(model_time/model_frequency) < 100000:
            #     # else:
            #     #     foo = np.where(np.array(files) == prefix + '_' + str(int(model_time/model_frequency))+'.nc')[0][0]
                
            #     foo = 0 
            #     time_step = np.floor(lidar_parameter['model_time']/lidar_parameter['model_frequency'])
            #     # print(time_step)
            #     # sys.exit()
            #     temp = Read_file.get_cm1_data(x,y,z,lidar_parameter['lidar_x'],lidar_parameter['lidar_y'],lidar_parameter['lidar_z'],lidar_parameter['files'][foo],time_step,lidar_parameter['clouds'],namelist['turb'],az=azim,el=elev)
            #     # temp = get_cm1_data(x,y,z,lidar_x,lidar_y,lidar_z,files[foo],clouds,namelist['turb'],az=azim,el=elev)

            # else:
            #     print('ERROR: Unknown model type specified')
            #     return [-999]
            
            # Now we get the vr that the lidar would observe assuming a gaussian pulse
                
            # print(temp_vr.shape)
            # sys.exit()
            # def gaussian_pulse(vr,pulse_width,gate_width,maximum_range,nyquist_velocity,cut_off,r_sample,
            #                    key, elev, namelist):
            

            # print(j)   

    
    
    
class Data:   
    def __init__(self):     

        pass
    def write_to_file(sim_obs,scan_key,lidar_time,model_time_key,model_time,scans,namelist,scan_number):
        
        # We don't want to append the file needs to be created the first time
        
        if ((namelist['append'] == 0) & (not os.path.exists(namelist['output_dir'] + namelist['outfile']))):
            
            fid = Dataset(namelist['output_dir'] + namelist['outfile'],'w')
            
            rr = np.arange(3e8*namelist['gate_width']*1e-9/4.0,namelist['maximum_range']*1000+1,3e8*namelist['gate_width']*1e-9/2.0)
            
            if namelist['dbs1'] == 1:
                dbs1_rr = np.arange(namelist['dbs_start_height'],namelist['dbs_end_height']
                                   +1,namelist['dbs_spacing'])/np.cos(np.deg2rad(90-scans[0][0,1]))
            
            if namelist['dbs2'] == 2:
                dbs2_rr = np.arange(namelist['dbs_start_height'],namelist['dbs_end_height']
                                   +1,namelist['dbs_spacing'])/np.cos(np.deg2rad(90-scans[1][0,1])) 
            
            if namelist['dbs3'] == 1:
                dbs3_rr = np.arange(namelist['dbs_start_height'],namelist['dbs_end_height']
                                   +1,namelist['dbs_spacing'])/np.cos(np.deg2rad(90-scans[2][0,1]))
            
            if namelist['dbs4'] == 1:
                dbs4_rr = np.arange(namelist['dbs_start_height'],namelist['dbs_end_height']
                                   +1,namelist['dbs_spacing'])/np.cos(np.deg2rad(90-scans[3][0,1]))
            
            
            r = fid.createDimension('range',len(rr))
            stare_num = fid.createDimension('stare_num',None)
            
            base_time = fid.createVariable('base_time','f8')
            base_time.long_name = 'Epoch time'
            if namelist['use_calendar'] == 1:
                base_time.units = 's since 1970/01/01 00:00:00 UTC'
                base_time[:] = (model_time[0] - datetime(1970,1,1)).total_seconds()
            else:
                base_time.units = 's from model start time'
                base_time[:] = namelist['start_time']
            
            stare = fid.createVariable('vertical_stare','f8',('stare_num','range',))
            stare.long_name = 'vertical stare radial velocity'
            stare.units = 'm/s'
            
            stare_time = fid.createVariable('vertical_stare_time','f8',('stare_num',))
            stare_time.long_name = 'lidar time of vertical stare rays'
            stare_time.units = 's from base_time'
            stare_time.comment1 = 'This time variable can be different from the model time since lidar times and model times do not match'
            
            
            stare_model_time = fid.createVariable('vertical_stare_mtime', 'f8', ('stare_num',))
            stare_model_time.long_name = 'model time from which the ray was obtained from for vertical stares'
            stare_time.units = 's from base_time'
            
            ranges = fid.createVariable('ranges','f8',('range',))
            ranges.long_name = 'distance of each gate in ray from lidar'
            ranges.units = 'm'
            ranges[:] = rr[:]
            
            if len(scans) >= 1:
                scan1_num = fid.createDimension('scan1_num',None)
                scan1_rays = fid.createDimension('scan1_rays',len(scans[0]))
                
                
                if namelist['dbs1'] == 1:
                    dbs1_r = fid.createDimension('dbs_range1',len(dbs1_rr))
                    scan1_vr = fid.createVariable('scan1_vr','f8',('scan1_num','scan1_rays','dbs_range1',))
                    scan1_sw = fid.createVariable('scan1_sw','f8',('scan1_num','scan1_rays','dbs_range1',))

                else:
                    scan1_vr = fid.createVariable('scan1_vr','f8',('scan1_num','scan1_rays','range',))
                    scan1_sw = fid.createVariable('scan1_sw','f8',('scan1_num','scan1_rays','range',))
                    
                scan1_vr.long_name = 'radial velocity from scan 1'
                scan1_vr.units = 'm/s'
                scan1_sw.long_name = 'spectrum width from scan 1'
                scan1_sw.units = 'm/s'
                
                
                scan1_time = fid.createVariable('scan1_time','f8',('scan1_num','scan1_rays',))
                scan1_time.long_name = 'lidar time of scan 1 rays'
                scan1_time.units = 's from base_time'
                scan1_time.comment1 = 'This time variable can be different from the model time since lidar times and model times do not match'
                
                scan1_model_time = fid.createVariable('scan1_mtime','f8',('scan1_num','scan1_rays',))
                scan1_model_time.long_name = 'model time from which the ray was obtained from for scan 1'
                scan1_model_time.units = 's from base_time'
                
                scan1_az = fid.createVariable('scan1_az','f8',('scan1_rays',))
                scan1_az.long_name = 'azimuths for each ray in scan 1'
                scan1_az.units = 'degrees'
                scan1_az[:] = scans[0][:,0]
                
                scan1_el = fid.createVariable('scan1_el','f8',('scan1_rays',))
                scan1_el.long_name = 'elevation for each ray in scan 1'
                scan1_el.units = 'degrees'
                scan1_el[:] = scans[0][:,1]
            
            if len(scans) >= 2 :
                scan2_num = fid.createDimension('scan2_num',None)
                scan2_rays = fid.createDimension('scan2_rays',len(scans[1]))
                
                if namelist['dbs2'] == 1:
                    dbs2_r = fid.createDimension('dbs_range2',len(dbs2_rr))
                    scan2 = fid.createVariable('scan2','f8',('scan2_num','scan2_rays','dbs_range2',))
                else:
                    scan2 = fid.createVariable('scan2','f8',('scan2_num','scan2_rays','range',))
                    
                scan2.long_name = 'radial velocity from scan 2'
                scan2.units = 'm/s'
                
                scan2_time = fid.createVariable('scan2_time','f8',('scan2_num','scan2_rays',))
                scan2_time.long_name = 'lidar time of scan 2 rays'
                scan2_time.units = 's from base_time'
                scan2_time.comment1 = 'This time variable can be different from the model time since lidar times and model times do not match'
                
                scan2_model_time = fid.createVariable('scan2_mtime','f8',('scan2_num','scan2_rays',))
                scan2_model_time.long_name = 'model time from which the ray was obtained from for scan 2'
                scan2_model_time.units = 's from base_time'
                
                scan2_az = fid.createVariable('scan2_az','f8',('scan2_rays',))
                scan2_az.long_name = 'azimuths for each ray in scan 2'
                scan2_az.units = 'degrees'
                scan2_az[:] = scans[1][:,0]
                
                scan2_el = fid.createVariable('scan2_el','f8',('scan2_rays',))
                scan2_el.long_name = 'elevation for each ray in scan 2'
                scan2_el.units = 'degrees'
                scan2_el[:] = scans[1][:,1]
                
            if len(scans) >= 3:
                scan3_num = fid.createDimension('scan3_num',None)
                scan3_rays = fid.createDimension('scan3_rays',len(scans[2]))
                
                if namelist['dbs3'] == 1:
                    dbs3_r = fid.createDimension('dbs_range3',len(dbs3_rr))
                    scan3 = fid.createVariable('scan3','f8',('scan3_num','scan3_rays','dbs_range3',))
                else:
                    scan3 = fid.createVariable('scan3','f8',('scan3_num','scan3_rays','range',))
                    
                scan3.long_name = 'radial velocity from scan 3'
                scan3.units = 'm/s'
                
                scan3_time = fid.createVariable('scan3_time','f8',('scan3_num','scan3_rays',))
                scan3_time.long_name = 'lidar time of scan 3 rays'
                scan3_time.units = 's from base_time'
                scan3_time.comment1 = 'This time variable can be different from the model time since lidar times and model times do not match'
                
                scan3_model_time = fid.createVariable('scan3_mtime','f8',('scan3_num','scan3_rays',))
                scan3_model_time.long_name = 'model time from which the ray was obtained from for scan 3'
                scan3_model_time.units = 's from base_time'
                
                scan3_az = fid.createVariable('scan3_az','f8',('scan3_rays',))
                scan3_az.long_name = 'azimuths for each ray in scan 3'
                scan3_az.units = 'degrees'
                scan3_az[:] = scans[2][:,0]
                
                scan3_el = fid.createVariable('scan3_el','f8',('scan3_rays',))
                scan3_el.long_name = 'elevation for each ray in scan 3'
                scan3_el.units = 'degrees'
                scan3_el[:] = scans[2][:,1]
            
            if len(scans) >= 4:
                scan4_num = fid.createDimension('scan4_num',None)
                scan4_rays = fid.createDimension('scan4_rays',len(scans[3]))
                
                if namelist['dbs4'] == 1:
                    dbs4_r = fid.createDimension('dbs_range4',len(dbs4_rr))
                    scan4 = fid.createVariable('scan4','f8',('scan4_num','scan4_rays','dbs_range4',))
                else:
                    scan4 = fid.createVariable('scan4','f8',('scan4_num','scan4_rays','range',))
                    
                scan4.long_name = 'radial velocity from scan 4'
                scan4.units = 'm/s'
                
                scan4_time = fid.createVariable('scan4_time','f8',('scan4_num','scan4_rays',))
                scan4_time.long_name = 'lidar time of scan 4 rays'
                scan4_time.units = 's from base_time'
                scan4_time.comment1 = 'This time variable can be different from the model time since lidar times and model times do not match'
                
                scan4_model_time = fid.createVariable('scan4_mtime','f8',('scan4_num','scan4_rays',))
                scan4_model_time.long_name = 'model time from which the ray was obtained from for scan 4'
                scan4_model_time.units = 's from base_time'
                
                scan4_az = fid.createVariable('scan4_az','f8',('scan4_rays',))
                scan4_az.long_name = 'azimuths for each ray in scan 4'
                scan4_az.units = 'degrees'
                scan4_az[:] = scans[3][:,0]
                
                scan4_el = fid.createVariable('scan4_el','f8',('scan4_rays',))
                scan4_el.long_name = 'elevation for each ray in scan 4'
                scan4_el.units = 'degrees'
                scan4_el[:] = scans[3][:,1]
                
            # Now add the global attributes. This is mostly namelist variables
            
            fid.LidarSim_version = '0.0.1'
            if namelist['model'] == 1:
                fid.model = 'WRF'
            elif namelist['model'] == 2:
                fid.model = 'FastEddy'
            elif namelist['model'] == 3:
                fid.model = 'NCAR LES'
            fid.model_output_frequency = str(namelist['model_frequency']) + ' seconds'
            fid.instantaneous_scan = str(namelist['instantaneous_scan'])
            fid.scans = str(namelist['number_scans'])
            fid.scan_file1 = namelist['scan_file1']
            fid.cc1 = str(namelist['cc1'] )
            fid.repeat1 = str(namelist['repeat1']) + ' seconds'
            fid.dbs1 = str(namelist['dbs1'])
            fid.scan_file2 = namelist['scan_file2']
            fid.cc2 = str(namelist['cc2'] )
            fid.repeat2 = str(namelist['repeat2']) + ' seconds'
            fid.dbs2 = str(namelist['dbs2'])
            fid.scan_file3 = namelist['scan_file3']
            fid.cc3 = str(namelist['cc3'] )
            fid.repeat3 = str(namelist['repeat3']) + ' seconds'
            fid.dbs3 = str(namelist['dbs3'])
            fid.scan_file4 = namelist['scan_file4']
            fid.cc4 = str(namelist['cc4'] )
            fid.repeat4 = str(namelist['repeat4']) + ' seconds'
            fid.dbs1 = str(namelist['dbs4'])
            fid.stare_length = str(namelist['stare_length'])
            fid.motor_az_speed = str(namelist['motor_az_speed']) + ' degrees/sec'
            fid.motor_el_speed = str(namelist['motor_el_speed']) + ' degrees/sec'
            fid.scan1_az_speed = str(namelist['scan1_az_speed']) + ' degrees/sec'
            fid.scan1_el_speed = str(namelist['scan1_el_speed']) + ' degrees/sec'
            fid.scan2_az_speed = str(namelist['scan2_az_speed']) + ' degrees/sec'
            fid.scan2_el_speed = str(namelist['scan2_el_speed']) + ' degrees/sec'
            fid.scan3_az_speed = str(namelist['scan3_az_speed']) + ' degrees/sec'
            fid.scan3_el_speed = str(namelist['scan3_el_speed']) + ' degrees/sec'
            fid.scan4_az_speed = str(namelist['scan4_az_speed']) + ' degrees/sec'
            fid.scan4_el_speed = str(namelist['scan4_el_speed']) + ' degrees/sec'
            fid.ray_accumulation_time = str(namelist['ray_time']) + ' seconds'
            fid.pulse_width = str(namelist['pulse_width']) + ' ns'
            fid.gate_width = str(namelist['gate_width']) + ' ns'
            fid.maximum_range = str(namelist['maximum_range']) + ' km'
            fid.model_sample_resolution = str(namelist['sample_resolution']) + ' m'
            fid.nyquist_velocity = str(namelist['nyquist_velocity'])
            fid.coordinate_type = str(namelist['coordinate_type'])
            if namelist['coordinate_type'] == 1:
                fid.lidar_lat = str(namelist['lidar_lat'])
                fid.lidar_lon = str(namelist['lidar_lon'])
            else:
                fid.lidar_x = str(namelist['lidar_x'])
                fid.lidar_y = str(namelist['lidar_y'])
            fid.lidar_alt = str(namelist['lidar_alt'])
            fid.use_calendar = str(namelist['use_calendar'])
            if namelist['use_calendar'] == 1:
                fid.start_date = model_time[0].strftime('%Y-%m-%d %H:%M:%S')
                fid.end_time = model_time[-1].strftime('%Y-%m-%d %H:%M:%S')
            else:
                fid.start_time = str(namelist['start_time'])
                fid.end_time = str(namelist['end_time'])
            fid.scan_number = str(-1)
            fid.close()
            
        
        fid = Dataset(namelist['output_dir'] + namelist['outfile'],'a')
        
        base_time = fid['base_time']
        
        stare = fid['vertical_stare']
        stare_time = fid['vertical_stare_time']
        stare_model_time = fid['vertical_stare_mtime']
        
        if len(scans) >= 1:
            scan1_vr = fid['scan1_vr']
            scan1_sw = fid['scan1_sw']
            scan1_time = fid['scan1_time']
            scan1_model_time = fid['scan1_mtime']
        
        if len(scans) >= 2:
            scan2 = fid['scan2']
            scan2_time = fid['scan2_time']
            scan2_model_time = fid['scan2_mtime']
        
        if len(scans) >= 3:
            scan3 = fid['scan3']
            scan3_time = fid['scan3_time']
            scan3_model_time = fid['scan3_mtime']
        
        if len(scans) >= 4:
            scan4 = fid['scan4']
            scan4_time = fid['scan4_time']
            scan4_model_time = fid['scan4_mtime']
        
        if namelist['instantaneous_scan'] == 1:
            foo = np.where(np.array(scan_key) == 0)[0]
            # print('foo= '+str(foo))
            # sys.exit()
            if len(foo > 0):
                temp_index1 = stare.shape[0]
                temp_index2 = stare.shape[0] + len(foo)
                stare[temp_index1:temp_index2,:] = np.array([sim_obs['vr'][x] for x in foo])[:]
                if namelist['use_calendar'] == 1:
                    stare_time[temp_index1:temp_index2] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                    stare_model_time[temp_index1:temp_index2] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                else:
                    stare_time[temp_index1:temp_index2] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
                    stare_model_time[temp_index1:temp_index2] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
            
            foo = np.where(np.array(scan_key) == 1)[0]
            if len(foo > 0):
                temp_index1 = scan1_vr.shape[0]
                scan1_vr[temp_index1,:,:] = np.array([sim_obs['vr'][x] for x in foo])[:]
                scan1_sw[temp_index1,:,:] = np.array([sim_obs['sw'][x] for x in foo])[:]

                if namelist['use_calendar'] == 1:
                    scan1_time[temp_index1,:] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                    scan1_model_time[temp_index1,:] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                else:
                    # print('foo= '+str(len(foo)))
                    scan1_time[temp_index1,:] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
                    scan1_model_time[temp_index1,:] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
                    # print('foo= '+str(np.array([model_time[model_time_key] - base_time[0]])))
                    # print('foo= '+str(np.array([model_time[model_time_key] - base_time[0]] * len(foo))))
            foo = np.where(np.array(scan_key) == 2)[0]
            if len(foo > 0):
                temp_index1 = scan2.shape[0]
                scan2[temp_index1,:,:] = np.array([sim_obs['vr'][x] for x in foo])
                if namelist['use_calendar'] == 1:
                    scan2_time[temp_index1,:] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                    scan2_model_time[temp_index1,:] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                else:
                    scan2_time[temp_index1,:] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
                    scan2_model_time[temp_index1,:] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
            
            foo = np.where(np.array(scan_key) == 3)[0]
            if len(foo > 0):
                temp_index1 = scan3.shape[0]
                scan3[temp_index1,:,:] = np.array([sim_obs['vr'][x] for x in foo])
                if namelist['use_calendar'] == 1:
                    scan3_time[temp_index1,:] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                    scan3_model_time[temp_index1,:] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                else:
                    scan3_time[temp_index1,:] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
                    scan3_model_time[temp_index1,:] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
            
            foo = np.where(np.array(scan_key) == 4)[0]
            if len(foo > 0):
                temp_index1 = scan4.shape[0]
                scan4[temp_index1,:,:] = np.array([sim_obs['vr'][x] for x in foo])
                if namelist['use_calendar'] == 1:
                    scan4_time[temp_index1,:] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                    scan4_model_time[temp_index1,:] = np.array([(model_time[model_time_key] - datetime(1970,1,1)).total_seconds() - base_time[0]] * len(foo))
                else:
                    scan4_time[temp_index1,:] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
                    scan4_model_time[temp_index1,:] = np.array([model_time[model_time_key] - base_time[0]] * len(foo))
                
        else:
            obs_vr = np.array(sim_obs['vr'])
            obs_sw = np.array(sim_obs['sw'])

            if scan_key[0] == 0:
                temp_index1 = stare.shape[0]
                temp_index2 = stare.shape[0] + len(scan_key)
                stare[temp_index1:temp_index2,:] = obs_vr[:,:]
                stare_time[temp_index1:temp_index2] = np.array(lidar_time) - base_time[0]
                if namelist['use_calendar'] == 1:
                    temp = np.array([(model_time[x] - datetime(1970,1,1)).total_seconds() for x in model_time_key])
                else:
                    temp = np.array([model_time[x] for x in model_time_key])
                stare_model_time[temp_index1:temp_index2] = temp - base_time[0]
            
            elif scan_key[0] == 1:
                temp_index1 = scan1_vr.shape[0]
                scan1_vr[temp_index1,:,:] = obs_vr[:,:]
                scan1_time[temp_index1, :] = np.array(lidar_time) - base_time[0]
                if namelist['use_calendar'] == 1:
                    temp = np.array([(model_time[x] - datetime(1970,1,1)).total_seconds() for x in model_time_key])
                else:
                    temp = np.array([model_time[x] for x in model_time_key])
                scan1_model_time[temp_index1,:] = temp - base_time[0]
            
            elif scan_key[0] == 2:
                temp_index1 = scan2.shape[0]
                scan2[temp_index1,:,:] = obs_vr[:,:]
                scan2_time[temp_index1, :] = np.array(lidar_time) - base_time[0]
                if namelist['use_calendar'] == 1:
                    temp = np.array([(model_time[x] - datetime(1970,1,1)).total_seconds() for x in model_time_key])
                else:
                    temp = np.array([model_time[x] for x in model_time_key])
                scan2_model_time[temp_index1,:] = temp - base_time[0]
            
            elif scan_key[0] == 3:
                temp_index1 = scan3.shape[0]
                scan3[temp_index1,:,:] = obs_vr[:,:]
                scan3_time[temp_index1, :] = np.array(lidar_time) - base_time[0]
                if namelist['use_calendar'] == 1:
                    temp = np.array([(model_time[x] - datetime(1970,1,1)).total_seconds() for x in model_time_key])
                else:
                    temp = np.array([model_time[x] for x in model_time_key])
                scan3_model_time[temp_index1,:] = temp - base_time[0]
            
            elif scan_key[0] == 4:
                temp_index1 = scan1_vr.shape[0]
                scan4[temp_index1,:,:] = obs_vr[:,:]
                scan4_time[temp_index1, :] = np.array(lidar_time) - base_time[0]
                if namelist['use_calendar'] == 1:
                    temp = np.array([(model_time[x] - datetime(1970,1,1)).total_seconds() for x in model_time_key])
                else:
                    temp = np.array([model_time[x] for x in model_time_key])
                scan4_model_time[temp_index1,:] = temp - base_time[0]
        
        fid.scan_number = str(scan_number)
        fid.close()


