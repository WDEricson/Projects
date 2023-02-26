#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:24:06 2022

@author: atmo650
"""

#-----------------------------------------------------------------------------------#
#-------------------------------Import and stuff------------------------------------#
#-----------------------------------------------------------------------------------#
#getting data and wrf-related functions
import xarray as xr
from netCDF4 import Dataset
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, interplevel,
                 cartopy_ylim, latlon_coords, disable_xarray, ALL_TIMES)
#plotting stuff
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, Normalize
from metpy.plots import Hodograph, SkewT, colortables
#working with data, numpy and MetPy functions
import numpy as np
import metpy
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.units import units
#Might need to run this if can't find PROJ_LIB by default...
#import osos.environ["PROJ_LIB"] = "C:\\Users\\d010r356\\AppData\\Local\\Continuum\\anaconda3\\Library\\share"; fixr
#-----------------------------------------------------------------------------------#
#-------------------------Get data--------------------------------------------------#
#-----------------------------------------------------------------------------------#
ncfile=Dataset('/Users/williamericson/Downloads/wrfout_control_v3')
#----------------------Plot a sounding with hodograph inset-------------------------#
#Use initial sounding in corner of domain to represent the pre-convective environment
time=0
xloc=10
yloc=10
#grab the variables at the selected time, xloc, yloc to create sounding
snding_pres=getvar(ncfile,'pressure',timeidx=time)[:,xloc,yloc]
snding_hght=getvar(ncfile,'z',timeidx=time)[:,xloc,yloc]
snding_tmpc=getvar(ncfile,'tc',timeidx=time)[:,xloc,yloc]
snding_dwpc=getvar(ncfile,'td',timeidx=time)[:,xloc,yloc]
#wind variables have extra steps...
wspd_wdir=getvar(ncfile,'wspd_wdir',timeidx=time)[:,:,xloc,yloc]
snding_wspd=wspd_wdir[0,:]
snding_wdir=wspd_wdir[1,:]*units.degrees
snding_uwnd,snding_vwnd=mpcalc.wind_components(snding_wspd,snding_wdir)
#use metpy functions to find the LCL
lcl_pressure,lcl_temperature=mpcalc.lcl(snding_pres[0],snding_tmpc[0],snding_dwpc[0
])
print('LCL level: '+str(int(lcl_pressure.magnitude))+' hPa and LCL temperature: '+str(int(lcl_temperature.magnitude))+' C')
#calculate the parcel profile using the surface [level=0] temperature and dewpoint
parcel_prof=mpcalc.parcel_profile(snding_pres,snding_tmpc[0],snding_dwpc[0])
parcel_prof=parcel_prof.metpy.convert_units('degC')
#add a point excactly at the lcl so the profile is complete and precise
parcel_T=np.append(parcel_prof.values,np.array(lcl_temperature.magnitude))
parcel_p=np.append(snding_pres.values,np.array(lcl_pressure.magnitude))
#sort profile to put output in descending pressure
sort_index=np.argsort(-parcel_p)
parcel_T,parcel_p=parcel_T[sort_index],parcel_p[sort_index]
#-------------Now that we have the parcel and environmental profile, plot it on 
SkewT
fig=plt.figure(figsize=(9,9),dpi=100)
#create SkewT then plot environmental conditions (i.e. the sounding)
skew=SkewT(fig,rotation=40)
skew.plot(snding_pres,snding_tmpc,'r',linewidth=3)
skew.plot(snding_pres,snding_dwpc,'g',linewidth=3)
skew.plot_barbs(snding_pres,snding_uwnd,snding_vwnd)
#can change skewT bounds (pressure/temperature) to fit profile better
skew.ax.set_ylim(1000,150)
skew.ax.set_xlim(-25,40)
#plot lifted surface parcel path and a dot at the LCL
skew.plot(lcl_pressure,lcl_temperature,'ko',markerfacecolor='black')
skew.plot(parcel_p,parcel_T,'k',linewidth=3)
#shade areas of CAPE and CIN
skew.shade_cin(np.array(snding_pres),np.array(snding_tmpc),np.array(parcel_prof))
skew.shade_cape(np.array(snding_pres),np.array(snding_tmpc),np.array(parcel_prof))
#add thermo diagram lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
#create a hodograph inset axes object 40% width and height of the in upper right
ax_hod=inset_axes(skew.ax,'40%','40%',loc=1)
h=Hodograph(ax_hod,component_range=30.)
h.add_grid(increment=10)
h.plot_colormapped(snding_uwnd,snding_vwnd,snding_wspd)
plt.show()
#Can calculate various convection metrics, an example of the function to find CAPE:
sbcape=metpy.calc.cape_cin(snding_pres,snding_tmpc,snding_dwpc,parcel_prof,which_lfc='bottom',which_el='top')
print('The surface based CAPE is '+str(int(sbcape[0].magnitude))+' J/kg')
#define jet colormap with white around 0 to help visualization
cmap=get_cmap('jet')
cmap_21=cmap(np.linspace(0, 1, 21))
cmap_21[9][0:3]=[1,1,1]
cmap_21[10][0:3]=[1,1,1]
cmap_21[11][0:3]=[1,1,1]
acmap=ListedColormap(cmap_21)
#-------------Get ready to plot x-y plots----------------#
time=1    #select time to plot
zlvl=2000 #select height (m) to plot
smf=2     #how much smoothing (0=none, 10=more) do you want to do?
#get the starttime based on selected time above
fcst_mins=getvar(ncfile,'xtimes',timeidx=time)
#subset variables to that time
pres=getvar(ncfile,'pressure',timeidx=time)  #pressure
hght=getvar(ncfile,'z',timeidx=time)         #height
rdbz=getvar(ncfile,'dbz',timeidx=time)       #simulated reflectivity
wwnd=getvar(ncfile,'wa',timeidx=time)        #vertical wind
#horizontal wind stuff has extra steps...
wspd_wdir=getvar(ncfile,'wspd_wdir',timeidx=time,units='m s-1')
wspd=wspd_wdir[0,:]*units.meter_per_second
wdir=wspd_wdir[1,:]*units.degrees
uwnd,vwnd=mpcalc.wind_components(wspd,wdir)
#for all variables, interpolate to the specified height (zlvl), smooth using a
#smoothing factor of smf, and find mean in domain so can calculate the anomalies
pres_z=interplevel(pres,hght,zlvl)
pres_z_mean=pres_z.mean()
pres_z_sm=mpcalc.smooth_gaussian(pres_z,smf)
pres_z_sm_mean=pres_z_sm.mean()
rdbz_z=interplevel(rdbz,hght,zlvl)
uwnd_z=interplevel(uwnd,hght,zlvl)*units.meter_per_second
vwnd_z=interplevel(vwnd,hght,zlvl)*units.meter_per_second
wwnd_z=interplevel(wwnd,hght,zlvl)
uwnd_z_mean=uwnd_z.mean()
vwnd_z_mean=vwnd_z.mean()
uwnd_z_sm=mpcalc.smooth_gaussian(uwnd_z,smf)
vwnd_z_sm=mpcalc.smooth_gaussian(vwnd_z,smf)
wwnd_z_sm=mpcalc.smooth_gaussian(wwnd_z,smf)
uwnd_z_sm_mean=uwnd_z_sm.mean()
vwnd_z_sm_mean=vwnd_z_sm.mean()
#For wind shear, find closest height to selected height and take difference around that
dz=2 #interval of levels to take the wind difference
shlvl=abs(snding_hght-zlvl).argmin()
dudz=(snding_uwnd[shlvl+dz]-snding_uwnd[shlvl-dz])/(snding_hght[shlvl+dz]-
snding_hght[shlvl-dz])
dvdz=(snding_vwnd[shlvl+dz]-snding_vwnd[shlvl-dz])/(snding_hght[shlvl+dz]-
snding_hght[shlvl-dz])
#Find shear dot w'
grad_w=mpcalc.gradient(wwnd_z_sm,deltas=(ncfile.DX,ncfile.DY)) #[0]-->ydir [1]-->xdir
sh_dot_w=dudz.values*grad_w[1]+dvdz.values*grad_w[0]
#---------------------------Plots--------------------------------------
#define distance grid in simple distance (km) coordinants
xx=pres_z.west_east*ncfile.DX/1000
yy=pres_z.south_north*ncfile.DX/1000
#set vector filter to reduce clutter
skipn=8
skipx,skipy=slice(skipn,500,skipn),slice(skipn,500,skipn)
#set limits of plot to zoom in on the convection
xlim1,xlim2=25,60
ylim1,ylim2=25,60
#-----Radar and full wind
fig=plt.figure(figsize=(12,9))
#colorfill reflectivity
cf=plt.pcolor(xx,yy,mpcalc.smooth_gaussian(rdbz_z*units.dimensionless,smf),cmap='gist_ncar',norm=Normalize(-20, 60))
plt.colorbar(cf,pad=.05)
#plot full wind vectors IN WHITE, use scale to change size of vectors...
plt.quiver(xx[skipx],yy[skipy],uwnd_z_sm[skipx,skipy],vwnd_z_sm[skipx,skipy],color=
'w',scale=100)
#include plot of environmental shear vector IN RED for p' reference
plt.quiver(xlim1+5,ylim1+5,dudz,dvdz,color='r',scale=0.05)
#various limits, labels, title
plt.xlim(xlim1,xlim2)
plt.ylim(ylim1,ylim2)
plt.gca().set_xlabel('X (km)',fontsize=18)
plt.gca().set_ylabel('Y (km)',fontsize=18)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_title(f'Reflectivity(dbz), Full Wind at {fcst_mins.data} minutes',fontsize=14)
plt.show()
#-----PLOT: w' and p', and full wind
fig=plt.figure(figsize=(12,9))
cmax=np.max([abs(wwnd_z.min().values),wwnd_z.max().values])
cf=plt.contourf(xx,yy,wwnd_z_sm,cmap=acmap,levels=np.linspace(-cmax,cmax,21))
#change contour intervals and range
cnum=8
cmax=np.max([abs((pres_z-pres_z_mean).min().values),(pres_z-
pres_z_mean).max().values])
cstart,cstop=cmax/cnum,cmax
thelvls=np.concatenate((np.linspace(-cstop,-
cstart,cnum),np.linspace(cstart,cstop,cnum)))
ax=plt.contour(xx,yy,pres_z_sm-pres_z_sm_mean,colors='k',levels=thelvls)
#plt.clabel(ax,fontsize=12,inline=0,fmt='%1.2f')  #if you want labels, uncomment...
plt.colorbar(cf,pad=.05)
#plot full wind vectors
plt.quiver(xx[skipx],yy[skipy],uwnd_z_sm[skipx,skipy],vwnd_z_sm[skipx,skipy],color=
'k',scale=100)
#include plot of environmental shear vector for p' reference
plt.quiver(xlim1+5,ylim1+5,dudz,dvdz,color='r',scale=0.08)
plt.xlim(xlim1,xlim2)
plt.ylim(ylim1,ylim2)
plt.gca().set_xlabel('X (km)',fontsize=18)
plt.gca().set_ylabel('Y (km)',fontsize=18)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_title(f'wwnd (color), press. pert. (contours), at {fcst_mins.data} minutes',fontsize=14)
plt.show()
#-----PLOT: p', shear dot w', and wind'
fig=plt.figure(figsize=(12,9))
cmax=np.max([abs((pres_z-pres_z_mean).min().values),(pres_z-
pres_z_mean).max().values])
cf=plt.contourf(xx,yy,pres_z_sm-pres_z_sm_mean,cmap=acmap,levels=np.linspace(-
cmax,cmax,21))
#change contour intervals and range
cnum=8
cmax=np.max([abs(sh_dot_w.min().magnitude),sh_dot_w.max().magnitude])
cstart,cstop=cmax/cnum,cmax
thelvls=np.concatenate((np.linspace(-cstop,-
cstart,cnum),np.linspace(cstart,cstop,cnum)))
ax=plt.contour(xx,yy,sh_dot_w,colors='k',levels=thelvls)
#plt.clabel(ax,fontsize=12,inline=0,fmt='%1.2f')  #if you want labels, uncomment...
plt.colorbar(cf,pad=.05)
#plot wind vector anomaly
vmax=np.max([abs((uwnd_z_sm-uwnd_z_sm_mean).min()),(uwnd_z_sm-
uwnd_z_sm_mean).max(),
             abs((vwnd_z_sm-vwnd_z_sm_mean).min()),(vwnd_z_sm-
vwnd_z_sm_mean).max()])
plt.quiver(xx[skipx],yy[skipy],
           uwnd_z_sm[skipx,skipy]-uwnd_z_sm_mean,vwnd_z_sm[skipx,skipy]-
vwnd_z_sm_mean,
           color='k',scale=5*vmax)
#include plot of environmental shear vector for p' reference
plt.quiver(xlim1+5,ylim1+5,dudz,dvdz,color='r',scale=0.08)
plt.xlim(xlim1,xlim2)
plt.ylim(ylim1,ylim2)
plt.gca().set_xlabel('X (km)',fontsize=18)
plt.gca().set_ylabel('Y (km)',fontsize=18)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_title(f'Press Pert. (colors), S dot w_p (contours), and wind anom at {fcst_mins.data} minutes',fontsize=14)
plt.show()