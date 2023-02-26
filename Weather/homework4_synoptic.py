#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:49:51 2022

@author: williamericson
"""

#-----------------------------------------------------------------------------------#
#-------------------------------Import and stuff------------------------------------#
#-----------------------------------------------------------------------------------#
#plotting stuff
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, Normalize
#Siphon downloads data
from siphon.catalog import TDSCatalog
#standard working with data
from datetime import datetime
import numpy as np
#MetPy to do meteorology stuff
from metpy.interpolate import cross_section
import metpy.calc as mpcalc
from metpy.units import units
from metpy.xarray import xr
#match strings with wildcards
import fnmatch
#too many annoying warnings (generally you shouldn't do this...)
import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------------#
#-------------------------------Define Functions------------------------------------#
#-----------------------------------------------------------------------------------#
#Since we plot almost the same thing everytime, just pass in colorfill values and atitle
def sutcliff_plots(ds,cfill,geo_lvl1,geo_lvl2,title):
  fig=plt.figure(figsize=(14,14))
  #setup area and put in states and coastlines
  crs=ds['Temperature_isobaric'].metpy.cartopy_crs
  ax=plt.subplot(111, projection=crs)
  ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.75)
  ax.add_feature(cfeature.STATES,linewidth=0.5)
  
  #colorfill something 
  ax_cbar=ax.contourf(ds['x'],ds['y'],1e5*cfill,levels=np.arange(-16,17,1),
                      extend='both',cmap=acmap)
  fig.colorbar(ax_cbar,fraction=0.025, pad=0.04)
  
  #contour 1000-500 hPa thickness
  thck_contour=ax.contour(ds['x'],ds['y'],(geo_lvl2-geo_lvl1)/10,levels=np.arange(400,650,6),linewidths=2,colors='dimgray',linestyles='dashed')
  thck_contour.clabel(thck_contour.levels,fontsize=18,colors='dimgray',inline=1,inline_spacing=6,fmt='%i',rightside_up=False,use_clabeltext=False)

  
  #contour 1000-hPa heights
  hght_contour=ax.contour(ds['x'],ds['y'],geo_lvl1,linewidths=3,colors='k')
  hght_contour.clabel(hght_contour.levels,fontsize=18,colors='k',inline=1,inline_spacing=6,fmt='%i',rightside_up=False,use_clabeltext=False)
  
  #map bounds in longitude/latitude
  ax.set_extent((-131,-68,21,54))
  
  #time name issue (time/time2), try one, then the other...
  try:
    ax.set_title(f'1000hPa height (contour), 1000-500hPa thickness (dashed), {title} (color) \n'
                 f'Valid at: {ds["time"].dt.strftime("%Y-%m-%d %H:%MZ").item()}',fontsize=16)
  except:
    ax.set_title(f'1000hPa height (contour), 1000-500hPa thickness (dashed), {title} (color) \n'
                 f'Valid at: {ds["time2"].dt.strftime("%Y-%m-%d %H:%MZ").item()}',fontsize=16)
  
#-----------------------------------------------------------------------------------#
#-------------------------Get data--------------------------------------------------#
#-----------------------------------------------------------------------------------#
#select the model initiation time
year=2021
month=11
day=12
hour=0
dt=datetime(year,month,day,hour) #pulls the initiation hour
#forecast hour
fhr=0 #selects which forecast to use (0 for analysis field)
#read NAM218 Data from THREDDS server at:
base_url='https://www.ncei.noaa.gov/thredds/catalog/model-nam218/' #June 2020-present
#create url of file from date using f-string and datetime formats
cat=TDSCatalog(f'{base_url}{dt:%Y%m}/{dt:%Y%m%d}/catalog.xml')
print(f'{base_url}{dt:%Y%m}/{dt:%Y%m%d}/catalog.xml')
#use Siphon to find the data
#first, convert List of Dataset to clean string to find forecast hour
all_names=cat.datasets.filter_time_range(dt,dt)
all_names=str(all_names)
all_names=all_names[1:-1]
str_list=str(all_names).split(', ')
thefcst=['*_0'+str(fhr).zfill(2)+'.grb2']
test=fnmatch.filter(str_list,thefcst[0])
index=str_list.index(test[0])
#select file to use for download
tempds=cat.datasets.filter_time_range(dt,dt)[index]
print('Filename: ' +str(tempds))
#use NetCDF Subset Service (NCSS) to define a subset from the query
ncss=tempds.subset()
query=ncss.query()
query.lonlat_box(north=60,south=10,east=320,west=200)
query.all_times()
query.add_lonlat()
query.accept('netcdf')
#ncss.metadata.variables #<--uncomment to see all available variables/metadata
query.variables('Geopotential_height_isobaric',
                'Temperature_isobaric',
                'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric',
                'Vertical_velocity_pressure_isobaric',
                'Relative_humidity_isobaric',
                'Pressure_surface')
#finally use the subset query to actually download the NetCDF data
data=ncss.get_data(query)
#put it into something MetPy can work with
datastore=xr.backends.NetCDF4DataStore(data)
ds=xr.open_dataset(datastore).metpy.parse_cf()
#there is only 1 time, so squeeze out the time dimension
ds=ds.metpy.parse_cf().squeeze()
#convert isobaric coordinants from Pa to hPa, probably a better way to do this...
save_attrs=ds.coords['isobaric1'].attrs
ds.coords['isobaric1']=ds.coords['isobaric1']/100*units.hPa
ds.coords['isobaric1'].attrs=save_attrs
ds.coords['isobaric1'].attrs['units']='hPa'
#include lat/lon in the coordinants
ds=ds.assign_coords(lat=ds.lat,lon=ds.lon)
#for plot, define jet colormap with white around 0
cmap=get_cmap('jet')
cmap_21=cmap(np.linspace(0, 1, 21))
cmap_21[9][0:3]=[1,1,1]
cmap_21[10][0:3]=[1,1,1]
cmap_21[11][0:3]=[1,1,1]
acmap=ListedColormap(cmap_21)
#-----------------------------------------------------------------------------------#
#-----------------------Do things with data-----------------------------------------#
#-----------------------------------------------------------------------------------#
#f as a funciton of y
f=mpcalc.coriolis_parameter(ds['lat'])
#f-plane (use mean f in domain to put as fo, alternatively can just do fo=1e-4)
fo=np.array(f).mean()
#select levels, usually 1000 and 500 hPa
lvl1=1000
lvl2=500
#get geopotential height for each level selected above
geo_lvl1=ds['Geopotential_height_isobaric'].sel(isobaric1=lvl1)
geo_lvl2=ds['Geopotential_height_isobaric'].sel(isobaric1=lvl2)
#smooth!
geo_lvl1=mpcalc.smooth_rectangular(geo_lvl1,(15,15),10)
geo_lvl2=mpcalc.smooth_rectangular(geo_lvl2,(15,15),10)
#get geostrophic winds at each level (Vg is vector, ug and vg are components)
Vg1=mpcalc.geostrophic_wind(geo_lvl1)
ug1,vg1=Vg1[0],Vg1[1]
#
Vg2=mpcalc.geostrophic_wind(geo_lvl2)
ug2,vg2=Vg2[0],Vg2[1]
#
#get thermal wind, which is just the difference of Vg at two levels
#
uT= ug2-ug1
vT= vg2-vg1
#
#get relative geostrophic vorticity
rel_vor1=mpcalc.vorticity(ug1,vg1)
rel_vor1=mpcalc.smooth_rectangular(rel_vor1,(15,15),10) #smooth!
rel_vor2=mpcalc.vorticity(ug2,vg2)
rel_vor2=mpcalc.smooth_rectangular(rel_vor2,(15,15),10) #smooth!
#get thermal vorticity
thermal_vorticity=rel_vor2-rel_vor1
thermal_vorticity=mpcalc.smooth_rectangular(thermal_vorticity,(15,15),10) #smooth!
#Calculate each term in Sutcliffe (RHS)
#Note: by definition, mpcalc.advection is negative!
sutA=(2/fo)*mpcalc.advection(rel_vor1,uT,vT)
sutA=mpcalc.smooth_rectangular(sutA,(15,15),10)
sutB=(1/fo)*mpcalc.advection(thermal_vorticity,uT,vT)
sutB=mpcalc.smooth_rectangular(sutB,(15,15),10)
sutC=(1/fo)*mpcalc.advection(f,uT,vT)
sutC=mpcalc.smooth_rectangular(sutC,(15,15),10)
#Add up all three terms on the RHS to get total
sutFull=sutA+sutB+sutC
#-----------------------------------------------------------------------------------#
#-----------------------Make plots of everything------------------------------------#
#-----------------------------------------------------------------------------------#
#--Since there are six plots, it's good to use a standard plotting function--#
cfill=rel_vor1
title='1000hPa Rel. Vorticity'
sutcliff_plots(ds,cfill,geo_lvl1,geo_lvl2,title)
cfill=thermal_vorticity
title='Thermal Vort.'
sutcliff_plots(ds,cfill,geo_lvl1,geo_lvl2,title)
cfill=sutA
title='Sutcliffe Term A'
sutcliff_plots(ds,cfill,geo_lvl1,geo_lvl2,title)
cfill=sutB
title='Sutcliffe Term B'
sutcliff_plots(ds,cfill,geo_lvl1,geo_lvl2,title)
cfill=sutC
title='Sutcliffe Term C'
sutcliff_plots(ds,cfill,geo_lvl1,geo_lvl2,title)
cfill=sutFull
title='Sutcliffe Full'
sutcliff_plots(ds,cfill,geo_lvl1,geo_lvl2,title)