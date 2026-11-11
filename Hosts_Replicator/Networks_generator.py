# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 09:56:47 2025

@author: stenm
"""
import numpy as np
import scipy.stats as ss
from tqdm import tqdm
#####
def Distrib(Kbar,mult=3,kappa=None):
    """Generate support kappa and random prob. vector Pk."""
    if kappa is None :
        n=np.random.randint(int(Kbar)+1,int(Kbar)*mult)
        kappa=list(range(1,n+1))
    n=len(kappa)
    Pk=np.random.random(n)
    Pk=Pk/np.sum(Pk)
    return kappa,Pk
def FKbar(kappa,Pk):
    return np.sum([kap*p for kap,p in zip(kappa,Pk)])
def Distribution_random(Cible,kappa=None,mult=3,eps=1e-10,**kwargs):
    """
    Adjust a random probability vector Pk on support kappa so that <k> = Cible.
    Iteratively shifts mass below/above the target mean until convergence.
    """

    kappa,Pk=Distrib(Cible,mult=mult,kappa=kappa)
    err=FKbar(kappa,Pk)-Cible
    U=[k for k in kappa if k<Cible]
    ncible=len(U)+1
    while np.abs(err)>eps:
        #print(err)
        if err<0 : Pk[ncible:]=Pk[ncible:]*(1+0.1*min(-err,0.5))
        if err>0 : Pk[:ncible]=Pk[:ncible]*(1+0.1*min(err,0.5))
        Pk=Pk/sum(Pk)
        err=FKbar(kappa,Pk)-Cible
    return kappa,Pk


def Distribution_three(kbar,ecart=1,p=1/3,**kwargs):
    """Produced kappa,Pk the probability law on  kappa=[kbar-ecart,kbar,kbar+ecart] 
    with probability Pk=[(1-p)/2,p,(1-p)/2]"""
    kappa=[kbar-ecart,kbar,kbar+ecart]
    Pk=[(1-p)/2,p,(1-p)/2]
    return kappa,Pk
def Distribution_bin(kbar,n=None,**kwargs):
    '''Binomial distribution with mean kbar'''
    if n is None : n=kbar*(np.random.randint(5)+2)
    p=(kbar-1)/n
    P=ss.binom.pmf(np.arange(n + 1),n,p)
    kappa = np.arange(1, n + 2)  
    return kappa.tolist(),P
def Distribution_unif(kbar,n=0,**kwargs):
    """uniform distribution between kbar-n and kbar+n"""
    kappa=[kbar-n+k for k in range(2*n+1)]
    Pk=[1/(2*n+1)]*(2*n+1)
    return kappa,Pk




def Gen_Distrib(kbar,num=1,distrib='random',**kwargs):
    """
   Generate 'num' networks. Each network is represented by:
     kappa : degree support
     PP    : degree distribution
     mom2  : second moment <k^2>
   distrib can be one of:
     'random', 'three', 'bin' or  'unif'
   kwargs contains optional parameters for the chosen distribution.
   """
    kbar=int(kbar)
    ListNet=[]
    # Extract optional parameters
    if distrib=='three' : 
          isP=kwargs.get('p',None)
          isecart=kwargs.get('ecart',None)
          distrib_fun=Distribution_three
    elif distrib=='bin': 
        isnmax=kwargs.get('n',None)
        distrib_fun=Distribution_bin
    elif distrib=='unif': 
        isn=kwargs.get('n',None)  
        distrib_fun=Distribution_unif
    else : distrib_fun=Distribution_random
    for _ in range(num):
        # Randomize missing parameters depending on distribution
        if distrib=='three' : 
            if isP is None : kwargs['p']=np.random.random()
            if isecart is None : kwargs['ecart']=np.random.randint(kbar)
        if distrib=='bin': 
            if isnmax is None : kwargs['n']=kbar+np.random.randint(3*kbar)
        if distrib=='unif':
            if isn is None :kwargs['n']=np.random.randint(kbar)
        # Generate distribution
        kappa,PP=distrib_fun(kbar,**kwargs)
        mom2=sum([pk*k**2 for pk,k in zip(PP,kappa)])
        ListNet+=[(kappa,PP,mom2)]
    return ListNet
def weights_classes(Lkbar, bins=(0,0.5,7,15), itera=1000):
    """
    Estimate empirical class weights for the statistic
        v = sqrt(k2bar - k1bar^2)
    for each kbar in Lkbar, using random network generation.
    """

    quantile = {}

    for kbar in Lkbar:
        V = np.empty(itera)  # store simulated values of v

        # simulate iter networks
        for i in  tqdm(range(itera), desc=f"gen. {itera} Networks for kbar={kbar}"):
            distrib = np.random.choice(['random', 'three', 'bin','unif'])
            kappa, P, mom2 = Gen_Distrib(kbar, distrib=distrib,mult=3)[0]
            mk=sum(k*p for k,p in zip(kappa,P))
            V[i] = np.sqrt(np.real(mom2- mk**2))

        # compute class weights
        Q = [np.mean((V >= bins[j]) & (V < bins[j+1]))
             for j in range(len(bins)-1)]
        Q.append(np.mean(V >= bins[-1]))  # last class

        quantile[kbar] = Q

    return quantile
### graphical representation of the distribution of a Network
import matplotlib.pyplot as plt
def bardistrib(kappa,P,ax=None,title='Network Distribution',**kwargs):
    """this function need matplotlib.pyplot"""
    if ax is None : fig,ax=plt.subplots()
    ax.set_title(title,fontsize=20)
    ax.set_xlabel(r'$\kappa$',fontsize=16)
    ax.bar(kappa,P,**kwargs)
    if len(kappa)==1:ax.set_xlim((0,2*kappa[0]))
