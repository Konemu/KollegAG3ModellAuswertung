#provide auxiliary functionality by loading specific packages
from netCDF4 import Dataset #provides functionality to read/write binary NetCDF data
import matplotlib.pyplot as plt #provides functionality for graphical representation of data
import matplotlib.colors as colors #provides functionality for optimizing plot colors
import numpy as np #provides functionality for scientific computing
import cartopy.crs as ccrs #provides functionality for geographic projections
from cartopy.util import add_cyclic_point #provides functionality for closing gaps at periodic boundaries

from cdo import * #provides python bindings for climate data operators (CDO)
cdo=Cdo() #towards easier calling of Cdo()
#for further information regarding CDO see https://code.mpimet.mpg.de/projects/cdo/

import os

rootdir = "~/kollegdata"
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))
