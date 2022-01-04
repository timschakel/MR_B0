# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 09:25:59 2022

@author: Tim
"""
from wad_qc.modulelibs import wadwrapper_lib
import numpy as np
import matplotlib.pyplot as plt

def sort_b0_series(dcmInfile):
    # for B0 data, we expect 2 sets of Dicoms, normal Modulus data and the B0 map itself
    
    keymapping = [
                 ("0040,9096,0040,9224","0028,1052"), # Real World Value Intercept -> Rescale Intercept
                 ("0040,9096,0040,9225","0028,1053"), # Real World Value Slope -> Rescale Slope
                 ]
       
    for series in range(len(dcmInfile)):
        for i in range(len(dcmInfile[series]._datasets)):
            if not "RescaleIntercept" in dcmInfile[series]._datasets[i]: # in wrong place define for some MR files
                dcmInfile[series]._datasets[i].RescaleIntercept = wadwrapper_lib.readDICOMtag(keymapping[0][0],dcmInfile[series],i)
                dcmInfile[series]._datasets[i].RescaleSlope = wadwrapper_lib.readDICOMtag(keymapping[1][0],dcmInfile[series],i)
            if dcmInfile[series]._datasets[i].RescaleIntercept == '': # old files use a triple nested private tag
                dcmInfile[series]._datasets[i].RescaleIntercept = dcmInfile[series]._datasets[i][0x20019000][0][0x20011068][0][0x00281052].value
            if dcmInfile[series]._datasets[i].RescaleSlope == '': # old files use a triple nested private tag
                dcmInfile[series]._datasets[i].RescaleSlope = dcmInfile[series]._datasets[i][0x20019000][0][0x20011068][0][0x00281053].value
        if dcmInfile[series]._datasets[0].ImageType[3] == 'M':
            dcmInfileM = dcmInfile[series]
            pixeldataInM = np.transpose(dcmInfile[series].get_pixel_array(),(1,2,0)) #permute for x,y,z matrix shape
        if dcmInfile[series]._datasets[0].ImageType[3] == 'B0':
            dcmInfileB0 = dcmInfile[series]
            pixeldataInB0 = np.transpose(dcmInfile[series].get_pixel_array(),(1,2,0)) #permute for x,y,z matrix shape
        
    #error if no B0 was found
    if 'dcmInfileB0' or 'pixeldataInB0' not in locals():
        raise Exception('[MRB0_sort_b0_series] No B0 data found!')
        
    return dcmInfileM, pixeldataInM, dcmInfileB0, pixeldataInB0


def create_spherical_mask(shape, voxel_spacing=(1.0,1.0,1.0), center=None, radius=None):
    # Create a spherical mask for height h, width w and depth d centered at center with given radius

    x,y,z = shape

    if center is None:  # use the middle of the image
        center = np.array(np.array(shape)/2,dtype=int)

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center, shape-center)

    X,Y,Z=(np.ogrid[:x, :y, :z])
    dist_from_center = np.sqrt(((X - center[0]) * voxel_spacing[0]) ** 2  +\
                       ((Y - center[1]) * voxel_spacing[1]) ** 2  +\
                       ((Z - center[2]) * voxel_spacing[2]) ** 2)

    mask = dist_from_center <= radius
    return mask

def calc_rms_stats(pixeldataInB0ppm,mask):
    # calc stats per slice
    rms_stats = np.empty(pixeldataInB0ppm.shape[2]+1)
    for slice in range(pixeldataInB0ppm.shape[2]):
        slicedata = pixeldataInB0ppm[:, :, slice] * mask[:, :, slice]
        rms_stats[slice] = np.sqrt(np.mean(slicedata**2))
        
    rms_stats[slice+1] = np.sqrt(np.mean(pixeldataInB0ppm[mask]**2))
    
    return rms_stats
    
def calc_percentile_stats(pixeldataInB0ppm,mask):
    # calc percentile stats
    percentiles = [5,25,50,75,95]
    perc_stats = np.empty([pixeldataInB0ppm.shape[2]+1,5])
    for slice in range(pixeldataInB0ppm.shape[2]):
        slicedata = pixeldataInB0ppm[:, :, slice] * mask[:, :, slice]
        if np.max(slicedata) == 0:
            perc_stats[slice,:] = [0,0,0,0,0]
        else:
            perc_stats[slice,:] = np.percentile(slicedata[slicedata != 0],percentiles)
        
    perc_stats[slice+1,:] = np.percentile(pixeldataInB0ppm[mask],percentiles)
    
    return perc_stats
    
def create_figure1(pixeldataInB0,phantom,acqdate,acqtime,scanori):
    # figure 1
    # display slices of masked fieldmap (in Hz), use p5 and p95 as clip range

    fig1data = pixeldataInB0 * phantom #masked image
    fig1min = np.percentile(fig1data[fig1data != 0],5)
    fig1max = np.percentile(fig1data[fig1data != 0],95)
    filename = 'B0_'+scanori+'_figure1.jpg'
    fig, axs = plt.subplots(ncols=5, nrows=1, sharey=True, sharex=True,
                            constrained_layout=True, figsize=(10,3))
    title = scanori +" "+ acqdate +" "+ acqtime
    fig.suptitle(title,fontsize=24)
    
    for slice in range(pixeldataInB0.shape[2]):
        ax = axs[slice]
        im = ax.imshow(pixeldataInB0[:, :, slice],cmap='hot',vmin=fig1min,vmax=fig1max)
        ax.set_title('Slice '+str(slice+1))
    
    fig.colorbar(im, shrink=0.6)
    fig.savefig(filename,dpi=160)
    return filename

def create_figure2(pixeldataInB0ppm,pixeldataInB0,phantom,dsv100,dsv200,dsv300,
                   dsv350,acqdate,acqtime,imaging_frequency,scanori):
    # figure 2
    # Display the analysis on the center slice
    # mask of phantom, fieldmap in Hz, PPM map
    # overlay the r=5,10,15,17.5 masks
    fig1data = pixeldataInB0 * phantom #masked image
    fig1min = np.percentile(fig1data[fig1data != 0],5)
    fig1max = np.percentile(fig1data[fig1data != 0],95)
    
    filename = 'B0_'+scanori+'_figure2.jpg'
    fig, axs = plt.subplots(ncols=4, nrows=2, sharey=True, sharex=True,
                            constrained_layout=True, figsize=(10,5))
    title = scanori +" "+ acqdate +" "+ acqtime + " results slice 3"
    fig.suptitle(title,fontsize=24)
    
    slice=2 # center slice
    ax = axs[0,0]
    im = ax.imshow(phantom[:, :, slice],cmap='gray')
    ax.set_title('Mask slice '+str(slice+1))
    
    ax = axs[0,1]
    im = ax.imshow(pixeldataInB0[:, :, slice],cmap='hot',vmin=fig1min,vmax=fig1max)
    ax.set_title('Fieldmap [Hz] slice  '+str(slice+1))
    fig.colorbar(im, ax=ax)
    
    ax = axs[0,2]
    im = ax.imshow(pixeldataInB0ppm[:, :, slice],cmap='hot',vmin=fig1min/imaging_frequency,vmax=fig1max/imaging_frequency)
    ax.set_title('Fieldmap [ppm] slice '+str(slice+1))
    fig.colorbar(im, ax=ax)
    
    ax = axs[0,3]
    ax.remove()
    
    ax = axs[1,0]
    im = ax.imshow(pixeldataInB0ppm[:, :, slice] * dsv100[:, :, slice],
                   cmap='hot',vmin=fig1min/imaging_frequency,vmax=fig1max/imaging_frequency)
    ax.set_title('r=5cm')
    
    ax = axs[1,1]
    im = ax.imshow(pixeldataInB0ppm[:, :, slice] * dsv200[:, :, slice],
                   cmap='hot',vmin=fig1min/imaging_frequency,vmax=fig1max/imaging_frequency)
    ax.set_title('r=10cm')
    
    ax = axs[1,2]
    im = ax.imshow(pixeldataInB0ppm[:, :, slice] * dsv300[:, :, slice],
                   cmap='hot',vmin=fig1min/imaging_frequency,vmax=fig1max/imaging_frequency)
    ax.set_title('r=15cm')
    
    ax = axs[1,3]
    im = ax.imshow(pixeldataInB0ppm[:, :, slice] * dsv350[:, :, slice],
                   cmap='hot',vmin=fig1min/imaging_frequency,vmax=fig1max/imaging_frequency)
    ax.set_title('r=17.5cm')
    fig.savefig(filename,dpi=160)
    return filename

def create_figure3(pixeldataInB0ppm,phantom,acqdate,acqtime,scanori):
    # figure 3
    # histogram of ppm of all slices (50 bins)
    # cumulative ppm histogram
    # every slice: mean ppm with min and max as errorbars
    fig3data = pixeldataInB0ppm * phantom #masked image
    nslices = pixeldataInB0ppm.shape[2]
    
    filename = 'B0_'+scanori+'_figure3.jpg'
    fig, axs = plt.subplots(ncols=3, nrows=1,
                            constrained_layout=True, figsize=(12,3))
    title = scanori +" "+ acqdate +" "+ acqtime
    fig.suptitle(title,fontsize=24)
    
    ax1 = axs[0]
    ax1.hist(fig3data[fig3data != 0].flatten(), bins=50)
    ax1.set(title='Histogram',xlabel='ppm',ylabel='number of pixels')
    
    ax2 = axs[1]
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax2.hist(fig3data[fig3data != 0].flatten(), bins=50, cumulative=True,histtype='step')
    ax2.set(title='Cumulative Histogram',xlabel='ppm',ylabel='number of pixels')
    
    ax3 = axs[2]
    x = np.arange(1,nslices+1)
    slicemedian=np.empty(nslices)
    sliceuperr=np.empty(nslices)
    slicelowerr=np.empty(nslices)
    
    for slice in range(nslices):
        slicedata = fig3data[:,:,slice]
        slicemedian[slice] = np.median(slicedata[slicedata != 0])
        slicelowerr[slice] = np.percentile(slicedata[slicedata != 0],5)
        sliceuperr[slice] = np.percentile(slicedata[slicedata != 0],95)
        
    asymmetric_error = [slicelowerr-slicemedian,slicemedian-sliceuperr]
    ax3.errorbar(x,slicemedian,yerr=asymmetric_error,fmt='o-')
    ax3.set(xlabel='slice number', ylabel='ppm',title='Median ppm +/- p5/95')
    
    fig.savefig(filename,dpi=160)    
    return filename
