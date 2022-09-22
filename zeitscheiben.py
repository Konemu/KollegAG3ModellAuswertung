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
plt.rcParams['figure.figsize'] = (12, 6)
plt.rc('font', family='serif')
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


def lin(x, a, b):
    return a*x + b


import os

rootdir = "/home/max/zeitscheiben/"
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
    if dir == "COSMOS-AO_PreIndustrial_100years" or dir == "COSMOS-ASO_PreIndustrial_100years":
        continue
    # determine model, epoch and co2
    model = dir.split("_")[0]
    epoch = dir.split("_")[1]

    fnpts = dir.split("_")[0:-2]

    co2 = dir.split("_")[-2]
    if co2[1] == "x":
        x = int(co2[0])
    elif co2[-1] == "m":
        x = float(co2[0:3])/280
    
    models.append(model + ", " + epoch + ", " + str(np.round(x,1)) + "xCO2")
    epoch_ref ="PreIndustrial"

    # set correct var names
    exp_file_name_template = rootdir + dir + "/MODEL_EPOCH_XxCO2_100years.nc" 
    ref_file_name_template = "/home/max/zeitscheiben/MODEL_PreIndustrial_100years/MODEL_PreIndustrial_100years.nc"
    #exp_file = exp_file_name_template.replace("MODEL",model).replace("EPOCH",epoch).replace("X",x)
    exp_file = rootdir + dir + "/" + dir + ".nc"
    ref_file = ref_file_name_template.replace("MODEL",model)
    
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
        #retval=cdo.yearmonmean(input = realm, output = outfile_yearmeans)
        #computation of time means of spatially resolved yearly data
        outfile_timmean=outfile_yearmeans.replace("data/","").replace('.nc','_timmean.nc')
        #retval=cdo.timmean(input = outfile_yearmeans, output = outfile_timmean)
        #computation of global mean of year average spatially resolved data
        outfile_yearmeans_fldmean=outfile_yearmeans.replace('.nc','_fldmean.nc')
        #retval=cdo.fldmean(input = outfile_yearmeans, output = outfile_yearmeans_fldmean)
        #computation of global mean of time average spatially resolved data
        outfile_timmean_fldmean=outfile_timmean.replace('.nc','_fldmean.nc')
        #retval=cdo.fldmean(input = outfile_timmean, output = outfile_timmean_fldmean)
        #computation of Arctic mean of year average spatially resolved data
        outfile_yearmeans_arctic_mean=outfile_yearmeans.replace('.nc','_arctic_mean.nc')
        #retval=cdo.fldmean(input = "-sellonlatbox,0,360,60,90 %s"%(outfile_yearmeans), output = outfile_yearmeans_arctic_mean)
        #computation of Arctic mean of time average spatially resolved data
        outfile_timmean_arctic_mean=outfile_timmean.replace('.nc','_arctic_mean.nc')
        #retval=cdo.fldmean(input = "-sellonlatbox,0,360,60,90 %s"%(outfile_timmean), output = outfile_timmean_arctic_mean)
        #computation of low latitude mean of year average spatially resolved data
        outfile_yearmeans_lowlat_mean=outfile_yearmeans.replace('.nc','_lowlat_mean.nc')
        #retval=cdo.fldmean(input = "-sellonlatbox,0,360,0,30 %s"%(outfile_yearmeans), output = outfile_yearmeans_lowlat_mean)
        #computation of low latitude mean of time average spatially resolved data
        outfile_timmean_lowlat_mean=outfile_timmean.replace('.nc','_lowlat_mean.nc')
        #retval=cdo.fldmean(input = "-sellonlatbox,0,360,0,30 %s"%(outfile_timmean), output = outfile_timmean_lowlat_mean)
        
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
    exp_tas_timmean=exp_file_hdl.variables['temp2'][:].squeeze() #read specific variable from the opened NetCDF file
    ref_tas_timmean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close() #close the NetCDF file
    ref_file_hdl.close()

    #load yearmean global average time series
    exp_file_hdl=Dataset(exp_files["yearmeans_fldmean"])
    ref_file_hdl=Dataset(ref_files["yearmeans_fldmean"])
    exp_tas_yearmeans_fldmean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_tas_yearmeans_fldmean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load time average global average
    exp_file_hdl=Dataset(exp_files["timmean_fldmean"])
    ref_file_hdl=Dataset(ref_files["timmean_fldmean"])
    exp_tas_timmean_fldmean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_tas_timmean_fldmean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load yearmean low latitude average
    exp_file_hdl=Dataset(exp_files["yearmeans_lowlat_mean"])
    ref_file_hdl=Dataset(ref_files["yearmeans_lowlat_mean"])
    exp_tas_yearmeans_lowlat_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_tas_yearmeans_lowlat_mean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load time average low latitude average
    exp_file_hdl=Dataset(exp_files["timmean_lowlat_mean"])
    ref_file_hdl=Dataset(ref_files["timmean_lowlat_mean"])
    exp_tas_timmean_lowlat_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_tas_timmean_lowlat_mean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load yearmean arctic average
    exp_file_hdl=Dataset(exp_files["yearmeans_arctic_mean"])
    ref_file_hdl=Dataset(ref_files["yearmeans_arctic_mean"])
    exp_tas_yearmeans_arctic_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_tas_yearmeans_arctic_mean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load time average arctic average
    exp_file_hdl=Dataset(exp_files["timmean_arctic_mean"])
    ref_file_hdl=Dataset(ref_files["timmean_arctic_mean"])
    exp_tas_timmean_arctic_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_tas_timmean_arctic_mean=ref_file_hdl.variables['temp2'][:].squeeze()

    exp_file_hdl.close()
    ref_file_hdl.close()

    #load yearmean antarctic average
    exp_file_hdl=Dataset(exp_files["yearmeans_antarctic_mean"])
    ref_file_hdl=Dataset(ref_files["yearmeans_antarctic_mean"])
    exp_tas_yearmeans_antarctic_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_tas_yearmeans_antarctic_mean=ref_file_hdl.variables['temp2'][:].squeeze()
    exp_file_hdl.close()
    ref_file_hdl.close()

    #load time average antarctic average
    exp_file_hdl=Dataset(exp_files["timmean_antarctic_mean"])
    ref_file_hdl=Dataset(ref_files["timmean_antarctic_mean"])
    exp_tas_timmean_antarctic_mean=exp_file_hdl.variables['temp2'][:].squeeze()
    ref_tas_timmean_antarctic_mean=ref_file_hdl.variables['temp2'][:].squeeze()
    exp_file_hdl.close()
    ref_file_hdl.close()

    #load geographic coordinates (for plotting geographic maps)
    #it is assumed that reference data and experiment data are on the same grid, hence coordinates are loaded only once
    exp_file_hdl=Dataset(exp_files["timmean"])
    ref_file_hdl=Dataset(ref_files["timmean"])
    if exp_file.find("ICON") == -1 and exp_file.find("MCM-UA-1-0") == -1:
        model_lon=exp_file_hdl.variables['lon'][:]
        model_lat=exp_file_hdl.variables['lat'][:]
    else:
        model_lon=exp_file_hdl.variables['longitude'][:]
        model_lat=exp_file_hdl.variables['latitude'][:]
    exp_file_hdl.close()
    ref_file_hdl.close()

    # calculate climate sensitivity metrics
    exp_tas_yearmeans_fldmean_dev = np.std(exp_tas_yearmeans_fldmean)#/np.sqrt(100)
    ref_tas_yearmeans_fldmean_dev = np.std(ref_tas_yearmeans_fldmean)#/np.sqrt(100)
    cs = (exp_tas_timmean_fldmean-ref_tas_timmean_fldmean)*2/x
    delta_cs = np.sqrt(exp_tas_yearmeans_fldmean_dev**2 + ref_tas_yearmeans_fldmean_dev**2)/np.sqrt(2/x)
    
    climate_sens.append(cs)
    climate_sens_err.append(delta_cs)

    exp_tas_yearmeans_arctic_mean_dev = np.std(exp_tas_yearmeans_arctic_mean)#/np.sqrt(100)
    ref_tas_yearmeans_arctic_mean_dev = np.std(ref_tas_yearmeans_arctic_mean)#/np.sqrt(100)
    exp_tas_yearmeans_lowlat_mean_dev = np.std(exp_tas_yearmeans_lowlat_mean)#/np.sqrt(100)
    ref_tas_yearmeans_lowlat_mean_dev = np.std(ref_tas_yearmeans_lowlat_mean)#/np.sqrt(100)
    aa = (exp_tas_timmean_arctic_mean-ref_tas_timmean_arctic_mean)/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean)
    delta_aa = np.sqrt((exp_tas_yearmeans_arctic_mean_dev/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean))**2
                    +(ref_tas_yearmeans_arctic_mean_dev/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean))**2
                    +(-(exp_tas_timmean_arctic_mean-ref_tas_timmean_arctic_mean)/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean)**2 * exp_tas_yearmeans_lowlat_mean_dev)**2
                    +((exp_tas_timmean_arctic_mean-ref_tas_timmean_arctic_mean)/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean)**2 * ref_tas_yearmeans_lowlat_mean_dev)**2)
    
    arctic_amp.append(aa)
    arctic_amp_err.append(delta_aa)

    exp_tas_yearmeans_antarctic_mean_dev = np.std(exp_tas_yearmeans_antarctic_mean)#/np.sqrt(100)
    ref_tas_yearmeans_antarctic_mean_dev = np.std(ref_tas_yearmeans_antarctic_mean)#/np.sqrt(100)
    exp_tas_yearmeans_lowlat_mean_dev = np.std(exp_tas_yearmeans_lowlat_mean)#/np.sqrt(100)
    ref_tas_yearmeans_lowlat_mean_dev = np.std(ref_tas_yearmeans_lowlat_mean)#/np.sqrt(100)
    aaa = (exp_tas_timmean_antarctic_mean-ref_tas_timmean_antarctic_mean)/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean)
    delta_aaa = np.sqrt((exp_tas_yearmeans_antarctic_mean_dev/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean))**2
                    +(ref_tas_yearmeans_antarctic_mean_dev/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean))**2
                    +(-(exp_tas_timmean_antarctic_mean-ref_tas_timmean_antarctic_mean)/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean)**2 * exp_tas_yearmeans_lowlat_mean_dev)**2
                    +((exp_tas_timmean_antarctic_mean-ref_tas_timmean_antarctic_mean)/(exp_tas_timmean_lowlat_mean-ref_tas_timmean_lowlat_mean)**2 * ref_tas_yearmeans_lowlat_mean_dev)**2)

    antarctic_amp.append(aaa)
    antarctic_amp_err.append(delta_aaa)

cst = []
aat = []
fig, ax = plt.subplots()
for pair in zip(climate_sens, arctic_amp, models, climate_sens_err, arctic_amp_err):
    if pair[2].split(",")[2].replace(" ","") == "1xCO2":
        continue
        fmt = "v"
    elif pair[2].split(",")[2].replace(" ","") == "1.2xCO2":
        fmt = "^"
    elif pair[2].split(",")[2].replace(" ","") == "1.4xCO2":
        fmt = "<"
    elif pair[2].split(",")[2].replace(" ","") == "1.6xCO2":
        fmt = ">"
    elif pair[2].split(",")[2].replace(" ","") == "2xCO2":
        fmt = "p"
    elif pair[2].split(",")[2].replace(" ","") == "2.1xCO2":
        fmt = "s"
    elif pair[2].split(",")[2].replace(" ","") == "3xCO2" or pair[2].split(",")[2].replace(" ","") == "3.0xCO2":
        fmt = "*"
    elif pair[2].split(",")[2].replace(" ","") == "4xCO2":
        fmt = "x"

    if pair[2].split(",")[1].replace(" ","") == "Maastrichtian":
        color = "blue"
    elif pair[2].split(",")[1].replace(" ","") == "Miocene":
        color = "green"
    elif pair[2].split(",")[1].replace(" ","") == "Pliocene":
        color = "orange"
    elif pair[2].split(",")[1].replace(" ","") == "Eocene":
        color = "red"
    elif pair[2].split(",")[1].replace(" ","") == "PreIndustrial":
        color = "cyan"
    elif pair[2].split(",")[1].replace(" ","") == "Oligocene":
        color = "gray"
    cst.append(pair[0])
    aat.append(pair[1])
    ax.errorbar(pair[0], pair[1], xerr=pair[3], yerr=pair[4], label=pair[2], fmt=fmt, markersize=8, capsize=5, ecolor="black", color=color)


parameters, cov = opt.curve_fit(lin, cst, aat)
x = np.linspace(0,10,100)
y = lin(x, parameters[0], parameters[1])
print(parameters[0], parameters[1])
ax.plot(x,y, "--", color="gray", label = "Lin. fit")

ax.set_xlabel("Climate sensitivity (K)")
ax.set_ylabel("Arctic amplification")
ax.set_ylim(0,3.5)
ax.set_xlim(0,10)
    
ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
fig.tight_layout()

fig.savefig("/home/max/code/KollegAG3ModellAuswertung/out_zeitscheiben/phasespace.pdf")

fig, ax = plt.subplots()
for pair in zip(arctic_amp, antarctic_amp, models, arctic_amp_err, antarctic_amp_err):
    if pair[2].split(",")[2].replace(" ","") == "1xCO2":
        continue
        fmt = "v"
    elif pair[2].split(",")[2].replace(" ","") == "1.2xCO2":
        fmt = "^"
    elif pair[2].split(",")[2].replace(" ","") == "1.4xCO2":
        fmt = "<"
    elif pair[2].split(",")[2].replace(" ","") == "1.6xCO2":
        fmt = ">"
    elif pair[2].split(",")[2].replace(" ","") == "2xCO2":
        fmt = "p"
    elif pair[2].split(",")[2].replace(" ","") == "2.1xCO2":
        fmt = "s"
    elif pair[2].split(",")[2].replace(" ","") == "3xCO2" or pair[2].split(",")[2].replace(" ","") == "3.0xCO2":
        fmt = "*"
    elif pair[2].split(",")[2].replace(" ","") == "4xCO2":
        fmt = "x"

    if pair[2].split(",")[1].replace(" ","") == "Maastrichtian":
        color = "blue"
    elif pair[2].split(",")[1].replace(" ","") == "Miocene":
        color = "green"
    elif pair[2].split(",")[1].replace(" ","") == "Pliocene":
        color = "orange"
    elif pair[2].split(",")[1].replace(" ","") == "Eocene":
        color = "red"
    elif pair[2].split(",")[1].replace(" ","") == "PreIndustrial":
        color = "cyan"
    elif pair[2].split(",")[1].replace(" ","") == "Oligocene":
        color = "gray"
    ax.errorbar(pair[0], pair[1], xerr=pair[3], yerr=pair[4], label=pair[2], fmt=fmt, markersize=8, capsize=5, ecolor="black", color=color)
    
x = np.linspace(0,5,100)
ax.plot(x,x, "--", color="gray", label = "AA = AAA")

ax.set_xlabel("Arctic amplification")
ax.set_ylabel("Antarctic amplification")
ax.set_ylim(0,4)
ax.set_xlim(0,4)
    
ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("/home/max/code/KollegAG3ModellAuswertung/out_zeitscheiben/phasespace2.pdf")

import csv
with open('/home/max/code/KollegAG3ModellAuswertung/out_zeitscheiben/res.txt', 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
    write.writerow(["Epoch/Model", "CS", "δCS", "AA", "δAA", "AAA", "δAAA"])
    write.writerow([models, climate_sens, climate_sens_err, arctic_amp, arctic_amp_err, antarctic_amp, antarctic_amp_err])


