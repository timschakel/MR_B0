#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
#
# Changelog:
#   20220104: Initial version
#
# /MRB0_wadwrapper.py -c Config/dcm_series/mr_b0.json -d TestSet/B0 -r results.json
from __future__ import print_function

__version__ = '20220104'
__author__ = 'tschakel'

import os
import sys

sys.path.insert(0, 'C:/Users/Tim/github/wadqc')
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib
from wad_qc.modulelibs import pydicom_series as dcmseries
import numpy as np

if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 

import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

try:
    import pydicom as dicom
except ImportError:
    import dicom
    
import MRB0_lib as B0

def logTag():
    return "[MRB0_wadwrapper] "

##### Real functions
def b0_series(data, results, action):
    """
    MRB0: Analyze B0 homogeneity measurements
    """
    
    # load the data
    dcmInfile = dcmseries.read_files(data.series_filelist[0], True, readPixelData=False, splitOnPosition=True)
    dcmInfileM, pixeldataInM, dcmInfileB0, pixeldataInB0 = B0.sort_b0_series(dcmInfile)
    
    # get the orientation
    seriesname = dcmInfileM.info.SeriesDescription
    if seriesname.startswith('t'):
        scanori = 'TRANSVERSAL'
    elif seriesname.startswith('c'):
        scanori = 'CORONAL'
    elif seriesname.startswith('s'):
        scanori = 'SAGITTAL'
    else:
        scanori = 'UNKNOWN'
        
    # do some bookkeeping
    imaging_frequency = dcmInfileM.info.ImagingFrequency
    pixeldataInB0ppm = pixeldataInB0/imaging_frequency
    
    # define volumes for statistics
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    pixelDims = (int(dcmInfileB0.info.Rows), int(dcmInfileB0.info.Columns), len(dcmInfileB0._datasets))
    
    # Load spacing values (in mm)
    gap = float(dcmInfileB0.info.SpacingBetweenSlices)
    pixelSpacing = (float(dcmInfileB0.info.PixelSpacing[0]), float(dcmInfileB0.info.PixelSpacing[1]),float(dcmInfileB0.info.SliceThickness)+gap)
    
    # find pixel coords of geometric center (0,0)
    # ImagePositionPatient gives the x,y,z coordinates of the center of the pixel in the upper lefthand corner
    x0 = np.round((0 - dcmInfileB0.info.ImagePositionPatient[0] + (pixelSpacing[0] / 2) ) / pixelSpacing[0])
    y0 = np.round((0 - dcmInfileB0.info.ImagePositionPatient[1] + (pixelSpacing[1] / 2) ) / pixelSpacing[1])
    z0 = np.round((0 - dcmInfileB0.info.ImagePositionPatient[2] + (pixelSpacing[2] / 2) ) / pixelSpacing[2])
    
    # define the masks for DSVs
    dsv100 = B0.create_spherical_mask(pixeldataInB0ppm.shape,voxel_spacing=(pixelSpacing[0],pixelSpacing[1],pixelSpacing[2]), center=(x0,y0,z0), radius=50)
    dsv200 = B0.create_spherical_mask(pixeldataInB0ppm.shape,voxel_spacing=(pixelSpacing[0],pixelSpacing[1],pixelSpacing[2]), center=(x0,y0,z0), radius=100)
    dsv300 = B0.create_spherical_mask(pixeldataInB0ppm.shape,voxel_spacing=(pixelSpacing[0],pixelSpacing[1],pixelSpacing[2]), center=(x0,y0,z0), radius=150)
    dsv350 = B0.create_spherical_mask(pixeldataInB0ppm.shape,voxel_spacing=(pixelSpacing[0],pixelSpacing[1],pixelSpacing[2]), center=(x0,y0,z0), radius=175)
    phantom = pixeldataInM > 200

    # calculate rms statistics within the different masks
    rms_dsv100 = B0.calc_rms_stats(pixeldataInB0ppm,dsv100)
    rms_dsv200 = B0.calc_rms_stats(pixeldataInB0ppm,dsv200)
    rms_dsv300 = B0.calc_rms_stats(pixeldataInB0ppm,dsv300)
    rms_dsv350 = B0.calc_rms_stats(pixeldataInB0ppm,dsv350)
    
    # calculate different percentile statistics for each slice
    perc_stats = B0.calc_percentile_stats(pixeldataInB0ppm,phantom)
    
    # create figures
    acqdate = dcmInfileM.info.StudyDate
    acqtime = dcmInfileM.info.StudyTime
    fname_fig1 = B0.create_figure1(pixeldataInB0, phantom, acqdate, acqtime, scanori)
    fname_fig2 = B0.create_figure2(pixeldataInB0ppm,pixeldataInB0,phantom,dsv100,dsv200,dsv300,
                                   dsv350,acqdate,acqtime,imaging_frequency, scanori)
    fname_fig3 = B0.create_figure3(pixeldataInB0ppm, phantom, acqdate, acqtime, scanori)
    
    # collect results
    results.addString("Orientation", scanori)
    results.addString("SeriesDescription", seriesname)
    results.addObject("Figure1",fname_fig1)
    results.addObject("Figure2",fname_fig2)
    results.addObject("Figure3",fname_fig3)
    
    reportkeyvals = []
    for slice in range(pixelDims[2]):
        # results for the different slices
        idname = "_slice"+str(slice+1)
        
        reportkeyvals.append( ("rms ppm d10"+idname,rms_dsv100[slice]) )
        reportkeyvals.append( ("rms ppm d20"+idname,rms_dsv200[slice]) )
        reportkeyvals.append( ("rms ppm d30"+idname,rms_dsv300[slice]) )
        reportkeyvals.append( ("rms ppm d35"+idname,rms_dsv350[slice]) )
        
        reportkeyvals.append( ("p5 ppm"+idname,perc_stats[slice,0]) )
        reportkeyvals.append( ("p25 ppm"+idname,perc_stats[slice,1]) )
        reportkeyvals.append( ("p50 ppm"+idname,perc_stats[slice,2]) )
        reportkeyvals.append( ("p75 ppm"+idname,perc_stats[slice,3]) )
        reportkeyvals.append( ("p95 ppm"+idname,perc_stats[slice,4]) )
    
    #whole phantom results
    reportkeyvals.append( ("rms ppm d10_phantom",rms_dsv100[pixelDims[2]]) )
    reportkeyvals.append( ("rms ppm d20_phantom",rms_dsv200[pixelDims[2]]) )
    reportkeyvals.append( ("rms ppm d30_phantom",rms_dsv300[pixelDims[2]]) )
    reportkeyvals.append( ("rms ppm d35_phantom",rms_dsv350[pixelDims[2]]) )
    
    reportkeyvals.append( ("p5 ppm_phantom",perc_stats[pixelDims[2],0]) )
    reportkeyvals.append( ("p25 ppm_phantom",perc_stats[pixelDims[2],1]) )
    reportkeyvals.append( ("p50 ppm_phantom",perc_stats[pixelDims[2],2]) )
    reportkeyvals.append( ("p75 ppm_phantom",perc_stats[pixelDims[2],3]) )
    reportkeyvals.append( ("p95 ppm_phantom",perc_stats[pixelDims[2],4]) )
    
    for key,val in reportkeyvals:
        results.addFloat(key, val)

def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database
    """

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)
    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)
    results.addDateTime('AcquisitionDateTime', dt) 
    
if __name__ == "__main__":
    data, results, config = pyWADinput()

    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)
        
        elif name == 'b0_series':
            b0_series(data, results, action)

    results.write()
