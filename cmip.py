#provide auxiliary functionality by loading specific packages
from netCDF4 import Dataset #provides functionality to read/write binary NetCDF data
import matplotlib.pyplot as plt #provides functionality for graphical representation of data
import matplotlib.colors as colors #provides functionality for optimizing plot colors
import numpy as np #provides functionality for scientific computing
import cartopy.crs as ccrs #provides functionality for geographic projections
from cartopy.util import add_cyclic_point #provides functionality for closing gaps at periodic boundaries

import scipy.optimize as opt

from cdo import * #provides python bindings for climate data operators (CDO)
cdo=Cdo() #towards easier calling of Cdo()
#for further information regarding CDO see https://code.mpimet.mpg.de/projects/cdo/

# nicer plots
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.rc('font', family='serif')
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


def lin(x, a, b):
    return a*x + b


import os

rootdir = "/home/max/cmip6/"
dataset_dirs = os.listdir(rootdir)

# result containers:
models = []
climate_sens = []
climate_sens_err = []
arctic_amp = []
arctic_amp_err = []
antarctic_amp = []
antarctic_amp_err = []

for dir in dataset_dirs:
    # determine model
    models.append(dir)
    model = dir

    time1 = ""
    time2 = ""
    flag = False
    
    for file in os.listdir(rootdir+"/"+dir):
        if file.find("abrupt") != -1 and not flag:
            flag = True
            time1 = file.split("_")[-1].replace(".nc","").replace(":Zone.Identifier","")
        if file.find("piControl") != -1 and flag:
            time2 = file.split("_")[-1].replace(".nc","").replace(":Zone.Identifier","")
    time1_num = int(time1.split("-")[1])
    time2_num = int(time2.split("-")[1])

    # set correct var names
    exp_file_name_temp2late = rootdir + dir + "/tas_Amon_MODEL_abrupt-4xCO2_TIME.nc" 
    ref_file_name_temp2late = rootdir + dir + "/tas_Amon_MODEL_piControl_TIME.nc"
    exp_file = exp_file_name_temp2late.replace("MODEL",model).replace("TIME",time1)
    ref_file = ref_file_name_temp2late.replace("MODEL",model).replace("TIME",time2)

    # generate data using cdo

    filename_dictionary={
    "yearmeans": "",
    "timmean": "",
    "yearmeans_fldmean": "",
    "timmean_fldmean": "",
    "yearmeans_arctic_mean": "",
    "timmean_arctic_mean": "",
    "yearmeans_arctic_mean": "",
    "timmean_arctic_mean": "",
    "yearmeans_lowlat_mean": "",
    "timmean_lowlat_mean": "",
    "yearmeans_arctic_mean": "",
    "timmean_arctic_mean": "",
    "yearmeans_antarctic_mean": "",
    "timmean_antarctic_mean": "",
    }
    exp_files=filename_dictionary.copy()
    ref_files=filename_dictionary.copy()
    for realm in exp_file,ref_file:
        #computation of year means of spatially resolved monthly data
        outfile_yearmeans=realm.replace("data/","").replace('.nc','_yearmonmean.nc')
        if realm == exp_file:
            retval=cdo.yearmonmean(input = ("-selyear,"+str(time1_num-99)+"/"+str(time1_num) + " %s")%(realm), output = outfile_yearmeans)
        else:
            retval=cdo.yearmonmean(input = ("-selyear,"+str(time2_num-99)+"/"+str(time2_num) + " %s")%(realm), output = outfile_yearmeans)
        #computation of time means of spatially resolved yearly data
        outfile_timmean=outfile_yearmeans.replace("data/","").replace('.nc','_timmean.nc')
        retval=cdo.timmean(input = outfile_yearmeans, output = outfile_timmean)
        #computation of global mean of year average spatially resolved data
        outfile_yearmeans_fldmean=outfile_yearmeans.replace('.nc','_fldmean.nc')
        retval=cdo.fldmean(input = outfile_yearmeans, output = outfile_yearmeans_fldmean)
        #computation of global mean of time average spatially resolved data
        outfile_timmean_fldmean=outfile_timmean.replace('.nc','_fldmean.nc')
        retval=cdo.fldmean(input = outfile_timmean, output = outfile_timmean_fldmean)
        #computation of Arctic mean of year average spatially resolved data
        outfile_yearmeans_arctic_mean=outfile_yearmeans.replace('.nc','_arctic_mean.nc')
        retval=cdo.fldmean(input = "-sellonlatbox,0,360,60,90 %s"%(outfile_yearmeans), output = outfile_yearmeans_arctic_mean)
        #computation of Arctic mean of time average spatially resolved data
        outfile_timmean_arctic_mean=outfile_timmean.replace('.nc','_arctic_mean.nc')
        retval=cdo.fldmean(input = "-sellonlatbox,0,360,60,90 %s"%(outfile_timmean), output = outfile_timmean_arctic_mean)
        #computation of low latitude mean of year average spatially resolved data
        outfile_yearmeans_lowlat_mean=outfile_yearmeans.replace('.nc','_lowlat_mean.nc')
        retval=cdo.fldmean(input = "-sellonlatbox,0,360,0,30 %s"%(outfile_yearmeans), output = outfile_yearmeans_lowlat_mean)
        #computation of low latitude mean of time average spatially resolved data
        outfile_timmean_lowlat_mean=outfile_timmean.replace('.nc','_lowlat_mean.nc')
        retval=cdo.fldmean(input = "-sellonlatbox,0,360,0,30 %s"%(outfile_timmean), output = outfile_timmean_lowlat_mean)
        
        #computation of ANTarctic mean of year average spatially resolved data
        outfile_yearmeans_antarctic_mean=outfile_yearmeans.replace('.nc','_antarctic_mean.nc')
        retval=cdo.fldmean(input = "-sellonlatbox,0,360,-90,-60 %s"%(outfile_yearmeans), output = outfile_yearmeans_antarctic_mean)
        #computation of ANTarctic mean of time average spatially resolved data
        outfile_timmean_antarctic_mean=outfile_timmean.replace('.nc','_antarctic_mean.nc')
        retval=cdo.fldmean(input = "-sellonlatbox,0,360,-90,-60 %s"%(outfile_timmean), output = outfile_timmean_antarctic_mean)
        
        if realm == exp_file:
            exp_files["yearmeans"]=outfile_yearmeans
            exp_files["timmean"]=outfile_timmean
            exp_files["yearmeans_fldmean"]=outfile_yearmeans_fldmean
            exp_files["timmean_fldmean"]=outfile_timmean_fldmean
            exp_files["yearmeans_lowlat_mean"]=outfile_yearmeans_lowlat_mean
            exp_files["timmean_lowlat_mean"]=outfile_timmean_lowlat_mean
            exp_files["yearmeans_arctic_mean"]=outfile_yearmeans_arctic_mean
            exp_files["timmean_arctic_mean"]=outfile_timmean_arctic_mean
            exp_files["yearmeans_antarctic_mean"]=outfile_yearmeans_antarctic_mean
            exp_files["timmean_antarctic_mean"]=outfile_timmean_antarctic_mean
        if realm == ref_file:
            ref_files["yearmeans"]=outfile_yearmeans
            ref_files["timmean"]=outfile_timmean
            ref_files["yearmeans_fldmean"]=outfile_yearmeans_fldmean
            ref_files["timmean_fldmean"]=outfile_timmean_fldmean
            ref_files["yearmeans_lowlat_mean"]=outfile_yearmeans_lowlat_mean
            ref_files["timmean_lowlat_mean"]=outfile_timmean_lowlat_mean
            ref_files["yearmeans_arctic_mean"]=outfile_yearmeans_arctic_mean
            ref_files["timmean_arctic_mean"]=outfile_timmean_arctic_mean
            ref_files["yearmeans_antarctic_mean"]=outfile_yearmeans_antarctic_mean
            ref_files["timmean_antarctic_mean"]=outfile_timmean_antarctic_mean

    # read generated netcdf files
    #load time average spatially resolved data
    exp_file_hdl=Dataset(exp_files["timmean"]) #open NetCDF file named in entry *timmean* of dictionary *exp_files*
    ref_file_hdl=Dataset(ref_files["timmean"])
    exp_temp2_timmean=exp_file_hdl.variables['temp2'][:].squeeze() #read specific variable from the opened NetCDF file
    ref_temp2_timmean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_precip_timmean=exp_file_hdl.variables['precip'][:].squeeze() # spatially resolved precip time avg for further tasks
    ref_precip_timmean=ref_file_hdl.variables['precip'][:].squeeze()

    exp_file_hdl.close() #close the NetCDF file
    ref_file_hdl.close()

    #load yearmean global average time series
    exp_file_hdl=Dataset(exp_files["yearmeans_fldmean"])
    ref_file_hdl=Dataset(ref_files["yearmeans_fldmean"])
    exp_temp2_yearmeans_fldmean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_temp2_yearmeans_fldmean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load time average global average
    exp_file_hdl=Dataset(exp_files["timmean_fldmean"])
    ref_file_hdl=Dataset(ref_files["timmean_fldmean"])
    exp_temp2_timmean_fldmean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_temp2_timmean_fldmean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load yearmean low latitude average
    exp_file_hdl=Dataset(exp_files["yearmeans_lowlat_mean"])
    ref_file_hdl=Dataset(ref_files["yearmeans_lowlat_mean"])
    exp_temp2_yearmeans_lowlat_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_temp2_yearmeans_lowlat_mean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load time average low latitude average
    exp_file_hdl=Dataset(exp_files["timmean_lowlat_mean"])
    ref_file_hdl=Dataset(ref_files["timmean_lowlat_mean"])
    exp_temp2_timmean_lowlat_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_temp2_timmean_lowlat_mean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load yearmean arctic average
    exp_file_hdl=Dataset(exp_files["yearmeans_arctic_mean"])
    ref_file_hdl=Dataset(ref_files["yearmeans_arctic_mean"])
    exp_temp2_yearmeans_arctic_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_temp2_yearmeans_arctic_mean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load time average arctic average
    exp_file_hdl=Dataset(exp_files["timmean_arctic_mean"])
    ref_file_hdl=Dataset(ref_files["timmean_arctic_mean"])
    exp_temp2_timmean_arctic_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_temp2_timmean_arctic_mean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load yearmean antarctic average
    exp_file_hdl=Dataset(exp_files["yearmeans_antarctic_mean"])
    ref_file_hdl=Dataset(ref_files["yearmeans_antarctic_mean"])
    exp_temp2_yearmeans_antarctic_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_temp2_yearmeans_antarctic_mean=ref_file_hdl.variables['temp2'][:].squeeze()
    exp_file_hdl.close()
    ref_file_hdl.close()

    #load time average antarctic average
    exp_file_hdl=Dataset(exp_files["timmean_antarctic_mean"])
    ref_file_hdl=Dataset(ref_files["timmean_antarctic_mean"])
    exp_temp2_timmean_antarctic_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_temp2_timmean_antarctic_mean=ref_file_hdl.variables['temp2'][:].squeeze()
    exp_file_hdl.close()
    ref_file_hdl.close()

    #load geographic coordinates (for plotting geographic maps)
    #it is assumed that reference data and experiment data are on the same grid, hence coordinates are loaded only once
    exp_file_hdl=Dataset(exp_files["timmean"])
    ref_file_hdl=Dataset(ref_files["timmean"])
    model_lon=exp_file_hdl.variables['lon'][:]
    model_lat=exp_file_hdl.variables['lat'][:]
    exp_file_hdl.close()
    ref_file_hdl.close()

    # calculate climate sensitivity metrics
    exp_temp2_yearmeans_fldmean_dev = np.std(exp_temp2_yearmeans_fldmean)#/np.sqrt(100)
    ref_temp2_yearmeans_fldmean_dev = np.std(ref_temp2_yearmeans_fldmean)#/np.sqrt(100)
    cs = (exp_temp2_timmean_fldmean-ref_temp2_timmean_fldmean)/2
    delta_cs = np.sqrt(exp_temp2_yearmeans_fldmean_dev**2 + ref_temp2_yearmeans_fldmean_dev**2)
    
    climate_sens.append(cs)
    climate_sens_err.append(delta_cs)

    exp_temp2_yearmeans_arctic_mean_dev = np.std(exp_temp2_yearmeans_arctic_mean)#/np.sqrt(100)
    ref_temp2_yearmeans_arctic_mean_dev = np.std(ref_temp2_yearmeans_arctic_mean)#/np.sqrt(100)
    exp_temp2_yearmeans_lowlat_mean_dev = np.std(exp_temp2_yearmeans_lowlat_mean)#/np.sqrt(100)
    ref_temp2_yearmeans_lowlat_mean_dev = np.std(ref_temp2_yearmeans_lowlat_mean)#/np.sqrt(100)
    aa = (exp_temp2_timmean_arctic_mean-ref_temp2_timmean_arctic_mean)/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean)
    delta_aa = np.sqrt((exp_temp2_yearmeans_arctic_mean_dev/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean))**2
                    +(ref_temp2_yearmeans_arctic_mean_dev/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean))**2
                    +(-(exp_temp2_timmean_arctic_mean-ref_temp2_timmean_arctic_mean)/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean)**2 * exp_temp2_yearmeans_lowlat_mean_dev)**2
                    +((exp_temp2_timmean_arctic_mean-ref_temp2_timmean_arctic_mean)/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean)**2 * ref_temp2_yearmeans_lowlat_mean_dev)**2)
    
    arctic_amp.append(aa)
    arctic_amp_err.append(delta_aa)

    exp_temp2_yearmeans_antarctic_mean_dev = np.std(exp_temp2_yearmeans_antarctic_mean)#/np.sqrt(100)
    ref_temp2_yearmeans_antarctic_mean_dev = np.std(ref_temp2_yearmeans_antarctic_mean)#/np.sqrt(100)
    exp_temp2_yearmeans_lowlat_mean_dev = np.std(exp_temp2_yearmeans_lowlat_mean)#/np.sqrt(100)
    ref_temp2_yearmeans_lowlat_mean_dev = np.std(ref_temp2_yearmeans_lowlat_mean)#/np.sqrt(100)
    aaa = (exp_temp2_timmean_antarctic_mean-ref_temp2_timmean_antarctic_mean)/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean)
    delta_aaa = np.sqrt((exp_temp2_yearmeans_antarctic_mean_dev/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean))**2
                    +(ref_temp2_yearmeans_antarctic_mean_dev/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean))**2
                    +(-(exp_temp2_timmean_antarctic_mean-ref_temp2_timmean_antarctic_mean)/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean)**2 * exp_temp2_yearmeans_lowlat_mean_dev)**2
                    +((exp_temp2_timmean_antarctic_mean-ref_temp2_timmean_antarctic_mean)/(exp_temp2_timmean_lowlat_mean-ref_temp2_timmean_lowlat_mean)**2 * ref_temp2_yearmeans_lowlat_mean_dev)**2)

    antarctic_amp.append(aaa)
    antarctic_amp_err.append(delta_aaa)

fig, ax = plt.subplots()
for pair in zip(climate_sens, arctic_amp, models, climate_sens_err, arctic_amp_err):
    ax.errorbar(pair[0], pair[1], xerr=pair[3], yerr=pair[4], label=pair[2], fmt="v", markersize=8, capsize=5, ecolor="black")

parameters, cov = opt.curve_fit(lin, climate_sens, arctic_amp)
x = np.linspace(0,10,100)
y = lin(x, parameters[0], parameters[1])
print(parameters[0], parameters[1])
ax.plot(x,y, "--", color="gray", label = "Lin. fit")

ax.set_xlabel("Climate sensitivity (K)")
ax.set_ylabel("Arctic amplification")
ax.set_ylim(0,3)
ax.set_xlim(0,10)
    
ax.legend(loc="lower right")
fig.tight_layout()

fig.savefig("/home/max/code/KollegAG3ModellAuswertung/out/phasespace.pdf")

fig, ax = plt.subplots()
for pair in zip(arctic_amp, antarctic_amp, models, arctic_amp_err, antarctic_amp_err):
    ax.errorbar(pair[0], pair[1], xerr=pair[3], yerr=pair[4], label=pair[2], fmt="v", markersize=8, capsize=5, ecolor="black")
    
x = np.linspace(0,5,100)
ax.plot(x,x, "--", color="gray")

ax.set_xlabel("Arctic amplification")
ax.set_ylabel("Antarctic amplification")
ax.set_ylim(0,5)
ax.set_xlim(0,5)
    
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("/home/max/code/KollegAG3ModellAuswertung/out/phasespace2.pdf")

import csv
with open('/home/max/code/KollegAG3ModellAuswertung/out/res.txt', 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
    write.writerow(["Epoch/Model", "CS", "δCS", "AA", "δAA", "AAA", "δAAA"])
    write.writerow([models, climate_sens, climate_sens_err, arctic_amp, arctic_amp_err, antarctic_amp, antarctic_amp_err])
