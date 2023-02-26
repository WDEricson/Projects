#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:50:28 2022

@author: williamericson
"""

#-----------------------------------------------------------------------------------#
#-------------------------------Import and stuff------------------------------------#
#-----------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------#
#create a simple, symmetric matrix corresponding to index val  
nx=100
ny= 100
X,Y=np.meshgrid(np.linspace(0,nx,nx),np.linspace(0,ny,ny))
#-----------------------------------------------------------------------------------#
#---------------------------Set up Vorticity Field----------------------------------#
#-----------------------------------------------------------------------------------#
#PART ONE
#Set the properties of the function
A=20
sigma=50
x_center=50
y_center=50
#expression for relative vorticity
vor= A*np.exp((-(((X-x_center)**2)+((Y-y_center)**2)))/(sigma))
#PART TWO
#Set all vorticity values to zero
#vor=np.zeros([nx,ny])
#Change a grid point(s) (i,j) from zero vorticity to some vorticity
#vor[45,50]=20
#vor[55,50]=20
#-----------------------------------------------------------------------------------#
#plot relative vorticity
fig=plt.subplots(figsize=(6,6),dpi=80)
plt.contourf(vor)
plt.gca().axis([0,nx,0,ny])
plt.gca().set_xlabel('X',fontsize=14)
plt.gca().set_ylabel('Y',fontsize=14)
plt.colorbar(fraction=0.03,pad=0.04)
#don't let a bad aspect ratio fool you!
plt.gca().set_aspect('equal', adjustable='box')
#set up matrices for psi, psi_temp, and the new/old residuals 
psi_temp=np.zeros([nx,ny])
psi=np.zeros([nx,ny])
residual_new=np.zeros([nx,ny])
residual_old=np.zeros([nx,ny])

#---Perform SOR
max_steps=500 #don't iterate more than this number of times
omega=1.94 #relaxation parameterr, helps convergence go quicker
#start loop for interations (k), and over the matrix (i,j)
for k in range(0,max_steps):
  for i in range(1,nx-1):
    for j in range(1,ny-1):
      psi_temp[i,j]=(psi[i,j+1]+psi[i,j-1]+psi[i+1,j]+psi[i-1,j]-vor[i,j])/4 
#calculate new psi field
      residual_new[i,j]=psi_temp[i,j]-psi[i,j]                               
      #find residuals
      psi[i,j]=psi[i,j]+omega*residual_new[i,j]                              
#adjust new psi field
  
  #find the difference of residuals and stop when that difference is small
  convergence_threshold=1e-3
  residual_dif=abs(residual_new-residual_old)
  if abs(residual_dif.sum())>convergence_threshold:
    print(abs(residual_dif.sum()))
  else:
    print('Done, converged at step: '+str(k))
    break
  #Just in case it didn't converge...
  if k==(max_steps-1):
    print('Did not converge before maximum steps reached.')
  
  #the new residuals become the old residuals
  residual_old=residual_new.copy()
  
'''
  #Plot each update to see how solution converges with each time step
  #I suggest uncommenting this section at least initially to watch how it converges
  #Especially fun for more complicated initial vorticity distributions
  fig=plt.subplots(figsize=(8,8),dpi=80)
  plt.contourf(psi)
  plt.gca().set_xlabel('X',fontsize=14)
  plt.gca().set_ylabel('Y',fontsize=14)
  #plt.colorbar(fraction=0.03,pad=0.04)
  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()
  '''
  
#Plot the psi field and associated winds, same as Hwk 2
#Calculate the gradient of psi with python function
gradient=np.array(np.gradient(psi))
dpsi_dx=gradient[1,:,:]
dpsi_dy=gradient[0,:,:]

print("go")
#Get u and v from the gradients of psi, double check signs...
u=dpsi_dx
v=dpsi_dy
#Calculate total kinetic energy in the domain (sum up kinetic energy at each grid point)


#KE= np.sum(((psi)**2)/2)
KE=np.sum(psi)
#Plot the retreved stream function and winds
fig=plt.subplots(figsize=(6,6),dpi=80)
plt.contourf(psi)
skip=(slice(None,None,5),slice(None,None,5)) #reduce clutter of vectors
#If vectors are too big or small, CHANGE THE SCALE!!
plt.quiver(X[skip],Y[skip],-dpsi_dy[skip],dpsi_dx[skip],color='black',scale=400)
plt.gca().axis([0,nx,0,ny])
plt.gca().set_xlabel('X',fontsize=14)
plt.gca().set_ylabel('Y',fontsize=14)
plt.colorbar(fraction=0.03,pad=0.04)
plt.title('Total KE='+str(int(KE)))
plt.gca().set_aspect('equal', adjustable='box')
