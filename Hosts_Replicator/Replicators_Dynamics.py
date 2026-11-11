# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 09:56:03 2025

@author: stenm
"""

#############
## Dynamic ##
#############
import numpy as np
import scipy.integrate as ode
from Hosts_Replicator.Replicators_generators import lamb
def replicator(t,z,Lambda=None):
    """definition of the replicator from a matrix Lambda and a state z"""
    if type(z) is list:
        z=np.array(z)
    if Lambda is None : Lambda=np.array([[]])
    if type(Lambda) is list : Lambda=np.array(Lambda)
    if np.shape(Lambda)!=(len(z),len(z)):
        print('Bad Lambda. Lambda=0 instead')
        Lambda=np.zeros((len(z),len(z)))

    zn=1/2*(z-np.abs(z))
    zp=z-zn
    z=zp-1e-15*zn
    z=z/sum(z)
    #print(z)
    Q=np.dot(z.T,np.dot(Lambda,z))
    L=np.dot(Lambda,z)
    return z*(L-Q)
def traject(T,LAMBDA,X0=None):
    """Computation of the dynamics for a Replicator system 
    using scipy.integrate.ode
    dt z=z ((LAMBDA z)-z.LAMBDA z)
    Simulation over [0,T]
    LAMBDA is the fitness matrix 
    X0 the initial state, stating homogeneous (1/n for each) by default.
    
    The output is the result of solve_ivp from ode
    """
    n=np.shape(LAMBDA)[0]
    if X0 is None : X0=np.ones(n)
    if type(X0) is list: X0=np.array(X0)
    if (X0<0).any() : 
        print('negativ entry')
        return None
    if (X0==0).all() : 
        print('no positiv entry')
        return  None
    X0=X0/np.sum(X0)
    def f_eval(t,X):
        return replicator(t,X,Lambda=LAMBDA)
    resultat = ode.solve_ivp(fun=f_eval, 
                             t_span=(0, T), 
                             y0=X0,
                             method='Radau',
                             dense_output=True,
                             #t_eval=np.arange(0,T,0.1),
                            )
    return resultat

def Final_Computation(n,T=100,X0=None,LL=None,**kwargs):
    """Computation of the dynamics of the replicator of size n over [0,T] 
    from an initial state X0
    
    There is various option in kwargs.
    #The Network definition one of the following :
    'CM'  the connection matrix 
    'Kappa' and 'P' directly the degree and there proportion
    
    #The global quantities (1 by default):
    r,gamma,rho,sigma.
    
    #The perturbations (0 by default)
    b : an array of size n     : perturbation on rho
    c1: an array of size n     : perturbation on gamma1 
    c2: an array of size (n,n) : perturbation on gamma2
    a : an array of size (n,n) : perturbation on sigma 
    """
    if LL is None : LL=lamb(n=n,**kwargs)
    res=traject(T,LL,X0)
    return res,LL


#############
## figures ##            
#############
import matplotlib.pyplot as plt

def Final_figure(res,nsteps=101,title='Replicator dynamics',ax=None,**kwargs):
    if ax is None : fig,ax=plt.subplots()
    title_kwargs = {}
    for key in ['fontsize', 'fontweight', 'color', 'family']:
        if key in kwargs:
            title_kwargs[key] = kwargs.pop(key)

    TT=np.linspace(0,res.t[-1],nsteps)
    sol=res.sol(TT)
    ax.plot(TT,sol.T,**kwargs)
    ax.set_title(title,**title_kwargs)
    ax.set_xlim((0,res.t[-1]))
    ax.set_ylim((0,1))
    return plt.gca()





