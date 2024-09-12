#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:12:48 2024

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
# import struct
import xarray as xr
# from netCDF4 import Dataset
# from scipy.interpolate import interp1d
# from scipy.special import erf
from argparse import ArgumentParser
from datetime import datetime, timedelta
import DLsim_Function as DLsim

# Read_file=DLsim.Read_file()

if __name__ == "__main__":

        
    #Create parser for command line arguments
    # parser = ArgumentParser()
    
    # parser.add_argument("namelist_file", help="Name of the namelist file (string)")
    # parser.add_argument("--output_dir", help="Path to output directory")
    # parser.add_argument("--debug", action="store_true", help="Set this to turn on the debug mode")
    
    # args = parser.parse_args()
    
    # namelist_file = args.namelist_file
    # output_dir = args.output_dir
    # debug = args.debug
        
    
    
    namelist_file =  'namelist_LES.input'
    output_dir = '/Users/hsiuweihsu/Resaerch/Metro_Weather_Project/LidarSim_Hsiu/output'
    debug = True
    
    if output_dir is None:
        output_dir = os.getcwd() + '/'
        
    if debug is None:
        debug = False
    
    print("-----------------------------------------------------------------------")
    print("Starting LidarSim")
    print("Output directory set to " + output_dir)
    
    # Read the namelist file
    namelist, scan_speeds = DLsim.Read_file.read_namelist(namelist_file)
    namelist['output_dir'] = output_dir
    if namelist['success'] != 1:
        print('>>> LidarSim FAILED and ABORTED <<<')
        print("-----------------------------------------------------------------------")
        sys.exit()
    
    # Read in the lidar scan files
    print('Reading in ' + str(namelist['number_scans']) + ' lidar scan files')
    # sys.exit()
    scans = []
    for i in range(namelist['number_scans']):
        try:
            scans.append(np.genfromtxt(namelist['scan_file'+ str(i+1)], delimiter= ' ',autostrip=True))
        except:
            print('ERROR: Something went wrong reading scan ' + str(i+1))
            print('>>> LidarSim FAILED and ABORTED <<<')
            print("-----------------------------------------------------------------------")
            sys.exit()
        if scans[i].shape[1] != 2:
            print('ERROR: The lidar scan files must be pairs of azimuth and elevation!. Offending scan: ' + str(i+1))
            print('>>> LidarSim FAILED and ABORTED <<<')
            print("-----------------------------------------------------------------------")
            sys.exit()
    # sys.exit()
    
    # Now set the scanning schedule. The end product is an array that will specify
    # the space and time of each beam for the scanning period
    
    if namelist['use_calendar'] == 1:
        start_time = datetime(namelist['start_year'],namelist['start_month'],namelist['start_day'],namelist['start_hour'],namelist['start_min'],namelist['start_sec'])
        end_time = datetime(namelist['end_year'],namelist['end_month'],namelist['end_day'],namelist['end_hour'],namelist['end_min'],namelist['end_sec'])
        
        model_time = np.arange(start_time,end_time+timedelta(seconds=namelist['model_frequency']),timedelta(seconds=namelist['model_frequency'])).astype(datetime)
        
        if namelist['model'] == 2:
            print('WARNING: FastEddy output is not typically output in calander time')
            print('         LidarSim might fail....')
            
    else:
        model_time = np.arange(namelist['start_time'],namelist['end_time']+namelist['model_frequency'],namelist['model_frequency'])
     
    
    az_el_coords, scan_key, model_time_key, lidar_time, scan_schedule = DLsim.Time.get_scan_timing(scans,model_time[0],model_time[-1],model_time,namelist['cced'],namelist['repeats'],
                                             namelist['stare_length'],scan_speeds,namelist['ray_time'],namelist['instantaneous_scan'],
                                             namelist['use_calendar'])    
    output_specific_schedule = 1
    # Declare the scanning schedule to the user
    if namelist['instantaneous_scan'] == 1:
        print('Running in instantaneous scan mode')
        print('The following scans will be collected instantaneously at every model time: ')
        if namelist['stare_length'] > 0:
            print('          > Vertical stare')
        for i in range(namelist['number_scans']):
            print('          > '  + namelist['scan_file' + str(i+1)])
    else:
        if output_specific_schedule == 1:
            f = open(output_dir + 'scan_schedule.txt','w')
            f.write('The scan schedule is as follows:\n')
            f.write('Scan                           Start                              End\n')
            f.write('____                        ____________                       ____________\n')
            if namelist['use_calendar'] == 1:
                for i in range(len(scan_schedule['scan'])):
                    f.write('{0:8s}'.format(scan_schedule['scan'][i]).rjust(8) + '                ' + datetime.utcfromtimestamp(scan_schedule['start'][i]).strftime('%Y-%m-%d %H:%M:%S') + '               ' + datetime.utcfromtimestamp(scan_schedule['end'][i]).strftime('%Y-%m-%d %H:%M:%S') + '\n')
            else:
                for i in range(len(scan_schedule['scan'])):
                    f.write('{0:8s}'.format(scan_schedule['scan'][i]).rjust(8) + '               ' + str(scan_schedule['start'][i]/3600.) + '               '  + str(scan_schedule['end'][i]/3600.)+'\n')
            f.close()
                    
        print('The scan schedule is as follows:')
        foo = np.where(namelist['cced'] == 1)[0]
        fah = np.where(namelist['cced'] == 0)[0]
        if len(foo) > 0:
            if namelist['stare_length'] > 0:
                print('          > Vertical stare - c/c')
            for i in range(len(foo)):
                print('          > '  + namelist['scan_file' + str(foo[i]+1)] + ' - c/c')
            for i in range(len(fah)):
                print('          > '  + namelist['scan_file' + str(fah[i]+1)] + ' Repeat every ' + str(namelist['repeat' + str(fah[i]+1)]/60.) + ' minutes')
        else:
            for i in range(namelist['number_scans']):
                print('          > '  + namelist['scan_file' + str(i+1)] + ' Repeat every ' + str(namelist['repeat' + str(i+1)]/60.) + ' minutes')
                print('          > Vertical stare at all other times')
    
    
    # Get the names of the model data
    
    if namelist['model_prefix'] == 'None':
        filename = namelist['model_dir'] + '*'
        dname =  namelist['model_dir']
    else:
        filename = namelist['model_dir'] + namelist['model_prefix'] + '*'
        dname = namelist['model_dir'] + namelist['model_prefix']
    
    # Now that we got the scan schedules we can start simulating lidar observations
    # For now, this is done serially, but after serial tests this will be done in parallel
    # using joblib. The output for LES models can be quite large, so data is LazyLoaded until
    # absolutely necessary to read into data. Often this will only be the data slices necessary
    # for the lidar simulation significantly reducing RAM usage.
    
    sim_obs = {}
    sim_obs['vr'] = []
    sim_obs['sw'] = []
    # sim_obs_sw = []
    # If the model data is in lat lons, change to x, y using info in the file
    # Right now, I am assuming that files will be like WRF output
    
    files = sorted(glob.glob(filename))
    
    if namelist['coordinate_type'] == 1:
        f = xr.open_dataset(files[0], decode_times=False)
        
        # LCC projection
        if f.MAP_PROJ == 1:
            wrf_proj = pyproj.Proj(proj='lcc',lat_1 = f.TRUELAT1, lat_2 = f.TRUELAT2, lat_0 = f.MOAD_CEN_LAT, lon_0 = f.STAND_LON, a = 6370000, b = 6370000)
            wgs_proj = pyproj.Proj(proj='latlong',datum='WGS84')
            transformer = pyproj.Transformer.from_proj(wgs_proj,wrf_proj)
        
        # Now transform the data
        e, n = transformer.transform(f.CEN_LON, f.CEN_LAT)
        dx,dy = f.DX, f.DY
        nx, ny = f.dims['west_east'], f.dims['south_north']
        x0 = -(nx-1) / 2. * dx + e
        y0 = -(ny-1) / 2. * dy + n
        xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
        lidar_x_proj, lidar_y_proj = transformer.transform(namelist['lidar_lon'], namelist['lidar_lat'])
        f.close()
    else:
        xx = None
        yy = None
        transformer = None
        lidar_x_proj = namelist['lidar_x']
        lidar_y_proj = namelist['lidar_y']
        
    # Now we are calling sim_obs. We have to keep track of which scans are written
    # to the file in case we need to run in append mode
    
    # These are counters for writing the output
    # We want to write to the output file everytime a scan is completed    
    sim_obs_begin = 0
    scan_number = 0
    
    # First check if this is an append run.
    if namelist['append'] == 1:
        
        # Make sure the files exists
        if os.path.exists(namelist['output_dir'] + namelist['outfile']):
            out = xr.open_dataset(namelist['output_dir'] + namelist['outfile'],decode_times=False)
            
            # Check to make sure instantaneous_scan parameter is the same in output file as namelist
            if namelist['instantaneous_scan'] != int(out.instantaneous_scan):
                print('ERROR: Instantaneous_scan parameter in output file does not match one set in namelist.')
                print('       Must abort!')
                sys.exit()
            
            # Get the scan number in the output file 
            scan_number = int(out.scan_number) + 1
            out.close()
            if namelist['instantaneous_scan'] == 0:
                sim_obs_begin = scan_schedule['start_index'][scan_number]
        else:
            print('Append mode was selcted, but ' +namelist['output_dir'] + namelist['outfile'] + ' does not exist.')
            print('A new output file will be created!')
            namelist['append'] = 0
    
    else:
        if os.path.exists(namelist['output_dir'] + namelist['outfile']):
            if namelist['clobber'] == 1:
                print(namelist['output_dir'] + namelist['outfile'] + ' exists and will be clobbered!')
                os.remove(namelist['output_dir'] + namelist['outfile'])
            else:
                print('ERROR:' + namelist['output_dir'] + namelist['outfile'] + ' exists and clobber is set to 0.')
                print('       Must abort to prevent file from being overwritten!')
                sys.exit()
                
   
    # We loop over all the model times        
    for i in range(len(model_time)):
        
        # Timing the simulation
        t0 = time.time()
        
        temp_lidar_x = lidar_x_proj - i*namelist['umove']*namelist['model_frequency']
        temp_lidar_y = lidar_y_proj - i*namelist['vmove']*namelist['model_frequency']
        
        # Find all the rays that need to be simulated from that output time
        foo = np.where(i == np.array(model_time_key))[0]
        # sys.exit()
        # Check to see if the simulations for this model time are all ready completed
        # For instantaneous scans this is as simple as checking the scan_number with i
        # but for regular scanning. For regular scanning, we need to check against the
        # index specified for the scan number that we are on.
        
        if namelist['instantaneous_scan'] == 1:
            if scan_number > i:
                bar = []
            else:
                bar = np.arange(len(foo)) 
        else:
            bar = np.where(foo >= sim_obs_begin)[0]
    
        # If the are rays to be simulated, we perform them
        if len(foo[bar]) > 0:
            print("Starting Simulations for " + str(model_time[i]))
            # sys.exit()
            lidar_parameter={'lidar_x':temp_lidar_x,
                             'lidar_y':temp_lidar_y,
                             'lidar_z':namelist['lidar_alt'],
                             'pulse_width':namelist['pulse_width'],
                             'gate_width':namelist['gate_width'],
                             'sample_resolution':namelist['sample_resolution'],
                             'maximum_range':namelist['maximum_range'],
                             'nyquist_velocity': namelist['nyquist_velocity'],
                             'coords':[az_el_coords[x] for x in foo[bar]],
                             'model_type':namelist['model'],
                             'model_time':model_time[i],
                             'model_step':namelist['model_timestep'],
                             'files':files,
                             'instantaneous_scan':namelist['instantaneous_scan'],
                             'prefix':dname,
                             'model_frequency':namelist['model_frequency'],
                             'nscl':namelist['ncar_les_nscl'],
                             'clouds':namelist['clouds'],
                             'scan_key':[scan_key[x] for x in foo[bar]],  
                             'sim_signal':namelist['sim_signal'],
                             'signal_file':namelist['signal_climo_file']
                             }
            
            # r_high = np.arange(1,(lidar_parameter['maximum_range']*1000)+1)
            # r = np.arange(3e8 * lidar_parameter['gate_width'] * 1e-9 / 4.0 ,lidar_parameter['maximum_range'] * 1000 + 1, 3e8 * lidar_parameter['gate_width'] * 1e-9 / 2.0)
            # r_high[None,:] - r[:,None]
            # def sim_observations(lidar_x, lidar_y, lidar_z, pulse_width, gate_width, sample_resolution, maximum_range,
            #                      nyquist_velocity, coords, model_type, model_time, model_step, files, instantaneous_scan,
            #                      prefix, model_frequency, nscl, clouds, scan_key,
            #                      sim_signal, signal_file, namelist, xx = None, yy = None, transform = None):
                
            
            # temp = DLsim.Signal.sim_observations(temp_lidar_x,temp_lidar_y,namelist['lidar_alt'], namelist['pulse_width'],
            #                     namelist['gate_width'], namelist['sample_resolution'], namelist['maximum_range'], namelist['nyquist_velocity'],
            #                     [az_el_coords[x] for x in foo[bar]],namelist['model'],model_time[i],namelist['model_timestep'],files, namelist['instantaneous_scan'],
            #                     dname,namelist['model_frequency'],namelist['ncar_les_nscl'],
            #                     namelist['clouds'],[scan_key[x] for x in foo[bar]],
            #                     namelist['sim_signal'],namelist['signal_climo_file'],
            #                     namelist, xx, yy, transformer)
            temp_vr,temp_sw,inter = DLsim.Signal.sim_observations(lidar_parameter,namelist, xx, yy, transformer)
            sys.exit()
            # c =3e8
            # pulse_width = namelist['pulse_width'] 
            # gate_width = namelist['gate_width']
            # r_high = np.arange(1,5001)
            # from scipy.special import erf

            # rwf = (1/(c*gate_width*1e-9))*\
            #       (erf((4*np.sqrt(np.log(2))*(r_high -3000)/(c*pulse_width*1e-9)) + np.sqrt(np.log(2))*gate_width/pulse_width)-\
            #        erf((4*np.sqrt(np.log(2))*(r_high -3000)/(c*pulse_width*1e-9)) - np.sqrt(np.log(2))*gate_width/pulse_width))
            
            
        # If not we move on to the next iteration since no new data  nothing will need
        # to be written to the file
            # sys.exit()
        else: 
            print("No simulation needed for " + str(model_time[i]))
            continue     
           
        if temp_vr[0] is int:
            print('ERROR: Something went wrong in observations simulation')
            sys.exit()
        else:
            # Put the simulated rays in a holding list until they are ready to be written
            sim_obs['vr'].extend(np.copy(temp_vr))
            sim_obs['sw'].extend(np.copy(temp_sw))

        t1 = time.time()
        print("Done with " + str(model_time[i]) + ' in ' + str(t1-t0) + ' secs')
        
        
        if namelist['instantaneous_scan'] == 1:
            t0 = time.time()
            DLsim.Data.write_to_file(sim_obs,scan_key[i],lidar_time[i],model_time_key[i],model_time,scans,namelist,i)

            t1 = time.time()
            print('Wrote scan ' + str(scan_number) + ' to output file in ' + str(t1-t0) + ' secs')
            
            scan_number = scan_number +1
            del sim_obs['vr'][:],sim_obs['sw'][:]
        
        else:
             # The current ending index of the scheduled rays 
            sim_obs_end = sim_obs_begin + len(sim_obs['vr'])-1
        
        
            # We will keep writing to the output file until there are no completed scans left
            # in the holding list  
    
            keep_writing = True
            while keep_writing:
            
                # We write to the file when all the rays for a scan are availble
                if ((scan_schedule['start_index'][scan_number] == sim_obs_begin) & (scan_schedule['end_index'][scan_number] <= sim_obs_end)):
                    t0 = time.time()
                    
                    DLsim.Data.write_to_file(sim_obs['vr'][0:scan_schedule['end_index'][scan_number]-sim_obs_begin+1],scan_key[sim_obs_begin:scan_schedule['end_index'][scan_number]+1],lidar_time[sim_obs_begin:scan_schedule['end_index'][scan_number]+1],
                              model_time_key[sim_obs_begin:scan_schedule['end_index'][scan_number]+1],model_time,scans,namelist,scan_number)
                
                    t1 = time.time()
    
                    print('Wrote scan ' + str(scan_number) + ' to output file in ' + str(t1-t0) + ' secs')
                
                
                    # Remove these rays from the holding list since they are written
                    # to file and update the counters
                    del sim_obs['vr'][0:scan_schedule['end_index'][scan_number]-sim_obs_begin+1]
                    sim_obs_begin = scan_schedule['end_index'][scan_number]+1
                    sim_obs_end = sim_obs_begin + len(sim_obs['vr'])-1
                    scan_number = scan_number + 1
                    if scan_number >= len(scan_schedule['start_index']):
                        keep_writing = False
                else:
                    # Stop writing
                    keep_writing = False

