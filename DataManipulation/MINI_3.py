#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
from scipy.integrate import solve_ivp

# https://en.wikipedia.org/wiki/Lorenz_96_model
def L96(t, x, N, F): 
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d

def get_observationsL96(start, stop, step, initial_state, N, F):
    # create a function that only takes timestep and state, with constants fixed
    fixed_lorenz = lambda t, state: L96(t, state, N, F)
    info = solve_ivp(fixed_lorenz, (start, stop), initial_state,
                     dense_output=True, method="RK45") # run a solver over the conditions
    sol = info.sol # get the solution function from the info object
    js = np.arange(start, stop, step) # get j's from start to stop with stepsize step
    truths = np.array([sol(j) for j in js]) # run solution over j's
    return truths


# In[23]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get observations from L96 from timesteps 0-30, stepsize 0.01,
# initial condition (0.01,0,0) and with rho, sigma, beta from the wikipedia page
N=5
F=8
x0 = F * np.ones(N) # equilibrium
x0[0] += 0.01  # small perturbation
truths = get_observationsL96(0, 30, 0.01, x0, N, F)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(truths[:, 0], truths[:, 1], truths[:, 2])
plt.draw()
plt.show()


# In[24]:


def generate_ensembleL96(n_members, sigma, m0, c0, start, stop, step, initial_state, N, F):
    # create a function that only takes timestep and state, with constants fixed
    fixed_lorenz = lambda t, state: L96(t, state, N, F)
    info = solve_ivp(fixed_lorenz, (start, stop), initial_state,
                     dense_output=True, method="RK45") # run a solver over the conditions
    sol = info.sol # get the solution function from the info object
    js = np.arange(start, stop, step) # get j's from start to stop with stepsize step

    # start with random ensemble
    vhat_0 = np.array([np.random.multivariate_normal(m0, c0) for _ in range(n_members)]).T
    vhat_at_timesteps = [vhat_0]
    for j in range(1, len(js)):
        vhat_j = np.zeros((N, n_members))
        for member_num in range(n_members):
            vhat_j[:, member_num] = sol(j-1) + np.random.multivariate_normal(np.zeros(N), sigma)
        vhat_at_timesteps.append(vhat_j)
    return vhat_at_timesteps


# In[25]:


N = 5
F = 8
sigma = np.random.randn(N, N)
m0 = np.zeros(N)
c0 = np.random.randn(N, N)
init_state = np.ones(N) * (F/2)
ens = generate_ensembleL96(10, sigma, m0, c0, 0, 40, 0.01, init_state, N, F)
print(len(ens))
print(ens[0].shape)
print(ens)


# In[26]:


#start second group
import numpy as np
import matplotlib.pyplot as plt
import sys

##L63 TRUTH GENERATOR##
from scipy.integrate import solve_ivp


# In[27]:


# https://en.wikipedia.org/wiki/Lorenz_system#Python_simulation
def lorenz63(t, current_state, rho, sigma, beta):
    x, y, z = current_state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def get_observationsL63(start, stop, step, initial_state, rho, sigma, beta):
    # create a function that only takes timestep and state, with constants fixed
    fixed_lorenz = lambda t, state: lorenz63(t, state, rho, sigma, beta)
    info = solve_ivp(fixed_lorenz, (start, stop), initial_state,
                     dense_output=True, method="RK45") # run a solver over the conditions
    sol = info.sol # get the solution function from the info object
    js = np.arange(start, stop, step) # get j's from start to stop with stepsize step
    truths = np.array([sol(j) for j in js]) # run solution over j's
    return truths


# In[28]:


# https://en.wikipedia.org/wiki/Lorenz_96_model
def L96(t, x, N, F): 
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d

def get_observationsL96(start, stop, step, initial_state, N, F):
    # create a function that only takes timestep and state, with constants fixed
    fixed_lorenz = lambda t, state: L96(t, state, N, F)
    info = solve_ivp(fixed_lorenz, (start, stop), initial_state,
                     dense_output=True, method="RK45") # run a solver over the conditions
    sol = info.sol # get the solution function from the info object
    js = np.arange(start, stop, step) # get j's from start to stop with stepsize step
    truths = np.array([sol(j) for j in js]) # run solution over j's
    return truths


# In[29]:


#grab a random noise point
def noise(v):
    mean = []
    mean.append(np.mean(v))
    cov_=[]
    cov=[]
    cov_.append(np.cov(v))
    cov.append(cov_)
    x = np.random.multivariate_normal(mean, cov)
    return x


# In[30]:


from mpl_toolkits.mplot3d import Axes3D

##DATA GENERATOR##
#v = the truth, n is the noise, h is the transformation matrix
def generate_point(v, N, ide):
    for i in range(1, N):
        n=noise(v[:])
        h=ide
        y=np.add(h, n)
    return y


# In[31]:


from enum import Enum
class H(Enum):
    threeVar = [[1,0,0],[0,1,0], [0,0,1]]
    oneVar1 = [[1,0,0], [0,0,0], [0,0,0]]
    oneVar2 = [[0,0,0], [0,1,0], [0,0,0]]
    oneVar3 = [[0,0,0], [0,0,0], [0,0,1]]
    twoVar12 = [[1,0,0], [0,1,0], [0,0,0]]
    twoVar13 = [[1,0,0], [0,0,0], [0,0,1]]
    twoVar23 = [[0,0,0], [0,1,0], [0,0,1]]


# In[32]:


#create data file L63
# Get observations from L63 from timesteps 0-40, stepsize 0.01, 
# initial condition (1,0,0) and with rho, sigma, beta from the wikipedia page
N=3
truths = get_observationsL63(0, 40, 0.01, (1.0, 0.0, 0.0), rho=28.0, sigma=10.0, beta=8.0/3.0)
np.set_printoptions(threshold=sys.maxsize)
outfile = open('truth.txt', 'w')
print(truths, file=outfile)
outfile.close()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(truths[:, 0], truths[:, 1], truths[:, 2])
plt.draw()
plt.savefig("L63.png")


ide=np.eye(N)
for i in range(1, N, 2):
    ide[i][i]=0;
outfile = open('L63.txt', 'w')
for i in truths:
    x = generate_point(i, N, ide)
    print(x, file = outfile)
    print(" ", file = outfile)
outfile.close()


# In[33]:


from mpl_toolkits.mplot3d import Axes3D

# Get observations from L96 from timesteps 0-30, stepsize 0.01,
# initial condition (0.01,0,0) and with rho, sigma, beta from the wikipedia page
N=10
F=8
x0 = F * np.ones(N) # equilibrium
x0[0] += 0.01  # small perturbation
truths = get_observationsL96(0, 30, 0.01, x0, N, F)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(truths[:, 0], truths[:, 1], truths[:, 2])
plt.draw()
plt.savefig("L96.png")

ide=np.eye(N)
for i in range(1, N, 2):
    ide[i][i]=0
outfile = open('L96.txt', 'w')
for i in truths:
    x = generate_point(i, N, ide)
    print(x, file = outfile)
    print(" ", file = outfile)
outfile.close()


# In[34]:


n=3; #Dimension of state space
m=2; #Dimension of observation space
N=10; #Number of ensemble members
H=np.eye(m,n); #Observation operator
dt = 0.1 #Time between observations
J = 1000 #Number of assimilation times
alpha=0.1
beta = 0.1
C0 = beta^2*np.eye(n)
Sigma = beta^2*np.eye(n)
Gamma = alpha^2*np.eye(m)


# In[ ]:


#Main Time Loop
for j in range(J):

    #Sample mean
    mhat = np.zeros((n,1))
    for k in range(N):
        mhat = mhat + ens[:,k]
    mhat = mhat/N

    #Sample covariance
    Chat = np.zeros(n)
    for k in range(N):
        covvec = ens[:,k]-mhat
        Chat = Chat + covvec*np.transpose(covvec)
    Chat = Chat/(N-1)

    


# In[ ]:





# In[ ]:





# In[ ]:




