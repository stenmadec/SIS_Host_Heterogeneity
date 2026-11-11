# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:03:48 2026

@author: stenm
"""

import numpy as np


########################################
##### Network #####
########################################
def Netstat_dict(CM, threshold=1e-5):
    """
    Degree distribution of a simple graph.

    Returns
    -------
    dict
        PN[k] = number of nodes of degree k.
    """
    CM = (CM + CM.T)/2
    np.fill_diagonal(CM, 0)

    PN = {}
    for row in CM:
        k = np.count_nonzero(np.abs(row) > threshold)
        PN[k] = PN.get(k, 0) + 1

    return PN
def Netstat(CM):
    """Computation of Kappa and P from a connectivity matrix CM"""
    PN=Netstat_dict(CM)
    Kappa=list(PN.keys())
    Kappa.sort()
    P0=[PN[k] for k in Kappa]
    N=sum(P0)
    P=[p/N for p in P0]
    return Kappa,P
def infer_size(**kwargs):
    """Infer the number of classes from vector parameters."""
    n = None

    for key in ('beta', 'r', 'gamma', 'sigma'):
        x = kwargs.get(key, None)

        if isinstance(x, (list, tuple)):
            m = len(x)

            if n is None:
                n = m
                n_key=key 
            elif m != n:
                raise ValueError(
                    f"All vector parameters must have the same length: "
                    f"'{n_key}' has length {n}, '{key}' has length {m}."
                    )

    return 1 if n is None else n
def init_data(**kwargs):
    """
   Construct model parameters from either:

   - a contact distribution q;
   - an adjacency matrix CM;
   - a degree distribution (Kappa, P).

   If q is not provided, the contact-biased degree distribution
   q_k = p_k k / <k> is computed automatically. Transmission rates
   are assumed proportional to degree, beta_k = rho k.

   Returns a dictionary containing q, beta, r, gamma and sigma.
   """
    def as_list(x, size=1):
        """Convert a scalar into a list of length size."""
        return list(x) if isinstance(x, (list, tuple)) else [x] * size

    lq = kwargs.get('q', None)

    if lq is None:

        CM = kwargs.get('CM', None)
    
        if CM is None and 'Kappa' not in kwargs and 'P' not in kwargs:
    
            n = infer_size(**kwargs)
    
            data = dict()
            lq = [1 / n] * n
            data['beta'] = as_list(kwargs.get('beta', 1.), n)
    
        else:
    
            if CM is None:
    
                Kappa = kwargs.get('Kappa', [2.])
    
                P = kwargs.get(
                    'P',
                    [1 / len(Kappa)] * len(Kappa)
                )
    
                if sum(P) != 1:
                    P = [pk / sum(P) for pk in P]
    
                if len(P) != len(Kappa):
                    P = [1 / len(Kappa) for _ in Kappa]
    
            else:
                Kappa, P = Netstat(CM)
    
            kbar = sum(p * k for p, k in zip(P, Kappa))
            rho = kwargs.get('beta', 2)
    
            data = dict()
            data['beta'] = [rho * k for k in Kappa]
    
            lq = [p * k / kbar for p, k in zip(P, Kappa)]
    
    else:
    
        data = dict()
        data['beta'] = as_list(kwargs.get('beta', 1.), len(lq))

    data['q'] = lq
    data['r'] = as_list(kwargs.get('r', 1.), len(lq))
    data['gamma'] = as_list(kwargs.get('gamma', 1.), len(lq))
    data['sigma'] = as_list(kwargs.get('sigma', 1.), len(lq))

    return data


def Netmain(Kappa,P):
    """Insert the existing degree Kappa and the frequency of each degree P. Return : 
    kbar = the mean degree of a node 
    k2bar=the mean square of a node"
    q    = the probability k*pk/kbar"""
    assert len(P)==len(Kappa), 'The network is badly defined : len(P)=/=len(Kappa)'
    if sum(P)!=1 : P=[pk/sum(P) for pk in P]## renormalisation of P if need
    kbar=sum(pk*k for pk,k in zip(P,Kappa))## Mean degree of the Network
    k2bar=sum(pk*k*k for pk,k in zip(P,Kappa))## Mean scare degree of the Network
    q=[pk*k/kbar for pk,k in zip(P,Kappa)]
    return kbar,k2bar,q
########################################
##### Equilibria of the SID system #####
########################################
def Phi(Theta,**data):
    R=[b/(r+g) for b,r,g in zip(data['beta'],data['r'],data['gamma'])]
    return sum(qk*rk/(1+rk*Theta) for qk,rk in zip(data['q'],R))



###########################################    
#####computation of the equilibria#########
###########################################
import scipy.optimize as opt


def Thetastar(**data):
    if Phi(0,**data)<=1 : return 0.
    def Phifind(Theta):
        return Phi(Theta,**data)-1
    return opt.fsolve(Phifind,0)[0]
def Skstar(Theta,r,beta,gamma,**kwargs):
    return (r+gamma)/(r+gamma+beta*Theta)
def Ikstar(Theta,r,beta,gamma,sigma,**kwargs):
    sk=Skstar(Theta,r,beta,gamma)
    R0k=(beta/(gamma+r))
    return (R0k*Theta*sk)/(1+sigma*R0k*Theta)
def Dkstar(Theta,r,beta,gamma,sigma,**kwargs):
    Rk=(beta/(gamma+r))
    return sigma*Rk*Theta*Ikstar(Theta,r,beta,gamma,sigma,**kwargs)

#####to remove
def Steady(**data):
    """Computation of the steady state for the SID system on a network. The nodes are classified by their degree k that we called the classes. For each classe k correspond a positive equilibria [sk*,ik*,dk*] whose sum is 1.
    This function return a dictionnary dicres whose keys are 
    Theta : The average Infetion strenght over the network 
    k : the possible degree of the nodes in this Network. 
    For each k,  dicres[k] is again a dictionnary with two keys :
    - p : the proportion of Nodes of degree k
    - state : the equilibria sk*,ik*,dk* for this class
    Note : There is at most one Endemic Equilibria (EE) over the  network. If the network has only one component, the EE exist everywhere or nowhere. If the EE does not exist we set the equilibria at [1,0,0] (the Endemic free Equilibria) for each class. """
    theta=Thetastar(**data)
    dicres={'theta':theta}
    #data=init_data(**kwargs)
    dicres['q']=data['q']
    for k,(r,g,b,s,q) in enumerate(zip(data['r'],
                                       data['gamma'],
                                       data['beta'],
                                       data['sigma'],
                                       data['q'])):
        degreek=dict()
        Sk=Skstar(theta,r,b,g)
        Ik=Ikstar(theta,r,b,g,s)
        Dk=Dkstar(theta,r,b,g,s)
        degreek['state']=[Sk,Ik,Dk]
        
        dicres[k]=degreek
    return dicres
def Xstar(result):
    """Isolation of the vector X=((ik,dk))_k in Kappa which is a steady state of the neutral model"""
    return np.array([result[k]['state'][1:] for k in result.keys() if type(k) is int]).flatten()


######################
## Replicator model ##
######################

def chi(**data):
    steady=Steady(**data)
        
    X=[(q/(r+g)) * (1-steady[k]['state'][0]
                     -0.5*steady[k]['state'][1]*
                     steady[k]['state'][2]/(1-steady[k]['state'][0]))
       for k,(r,g,b,s,q) in enumerate(zip(data['r'],
                                          data['gamma'],
                                          data['beta'],
                                          data['sigma'],
                                          data['q']))]
                    
    return 1/sum(X)
def Xi1(**data):
    steady=Steady(**data)
    x=sum(q*steady['theta']*s/(r+g) for k,(r,g,b,s,q) in enumerate(zip(data['r'],
                                              data['gamma'],
                                              data['beta'],
                                              data['sigma'],
                                              data['q'])))
    return x*chi(**data)
def Xi2(**data):
    steady=Steady(**data)
    x=sum(q*steady[k]['state'][1]*(1-0.5*steady[k]['state'][2]/(1-steady[k]['state'][0]))/(r+g)
                                   for k,(r,g,q) in enumerate(zip(data['r'],
                                              data['gamma'],
                                              data['q'])))
    return x*chi(**data)
def Xi3(**data):
    steady=Steady(**data)
    x=sum(q*steady[k]['state'][2]/(r+g) for k,(r,g,q) in enumerate(zip(data['r'],
                                              data['gamma'],
                                              data['q'])))
    return x*chi(**data)
def Xi4(**data):
    steady=Steady(**data)
    x=sum(q*steady['theta']
          *s*(b/(r+g))
          *steady[k]['state'][1]
          for k,(r,g,b,s,q) in enumerate(
                  zip(
                      data['r'],
                      data['gamma'],
                      data['beta'],
                      data['sigma'],
                      data['q']
                      )
                  )
          )
    return x*chi(**data)
def Xi5(**data):
    steady=Steady(**data)
    x = sum(
        q 
        * (b / (r + g))
        * steady[k]['state'][1]
        * steady[k]['state'][2] / (steady[k]['state'][1]+steady[k]['state'][2])
        for k, (r, g, b, q) in enumerate(
            zip(
                data['r'],
                data['gamma'],
                data['beta'],
                data['q']
            )
        )
    )                                                                             
    return (1/2)*x*chi(**data)* steady['theta']
def mu(**data):
    steady=Steady(**data)
    lq=steady.pop('q')
    _=steady.pop('theta')
    hh=[q*(b/(r+g))
    *steady[k]['state'][1]
    *steady[k]['state'][2]
    /(steady[k]['state'][1]+steady[k]['state'][2]) for k, (r, g, b, s, q) in enumerate(
        zip(
            data['r'],
            data['gamma'],
            data['beta'],
            data['sigma'],
            lq
        )
    )]
    lmu=[steady[k]['state'][1]/steady[k]['state'][2] for k in steady.keys()]
    x=sum(h*y for h,y in zip(hh,lmu))/sum(hh)
    return x*chi(**data)
    
    
def lamb(**kwargs):
    """
    Compute the Lambda matrix governing the replicator dynamics.

    Parameters
    ----------
    n : int, optional
        Number of strains. Used only when none of the variation parameters
        ('b', 'c1', 'c2', 'a') are provided. Default is 1.

    b : array_like, shape (n,), optional
        Deviations from neutrality in transmission rates.

    c1 : array_like, shape (n,), optional
        Deviations from neutrality in single-colonization clearance rates.

    c2 : array_like, shape (n, n), optional
        Deviations from neutrality in co-colonization clearance rates.

    a : array_like, shape (n, n), optional
        Deviations from neutrality in altered susceptibilities to
        co-colonization.

    w : array_like, shape (n, n), optional
        Deviations from neutrality in transmission priority effects.

    **kwargs :
        Additional keyword arguments passed to `init_data`, and subsequently
        to the coefficient functions `Xi1`--`Xi5` and `mu`.

    Returns
    -------
    ndarray, shape (n, n)
        Lambda matrix whose entries determine the pairwise invasion fitnesses
        appearing in the replicator equation.

    Notes
    -----
    Any omitted variation parameter is assumed to be identically zero,
    corresponding to neutrality for the associated trait.
    """
    for key in ('b', 'c1', 'c2', 'a','w'):
        if key in kwargs:
            arr = np.asarray(kwargs[key])
    
            if arr.ndim == 1:
                n = len(arr)
            elif arr.ndim == 2:
                n = arr.shape[0]
    
            break
    else:
        n = kwargs.pop('n',1)  # erreur naturelle si absent
    data=init_data(**kwargs)
    varb  = kwargs.pop('b',  np.zeros(n))
    varg1 = kwargs.pop('c1', np.zeros(n))
    varg2 = kwargs.pop('c2', np.zeros((n, n)))
    vara  = kwargs.pop('a',  np.zeros((n, n)))
    varom= kwargs.pop('w',np.zeros((n,n)))

    LAMB = (
        Xi1(**data) * (varb[:, None] - varb[None, :])
        + Xi2(**data) * (varg1[None, :] - varg1[:, None])
        + Xi3(**data) * (
            np.diag(varg2)[None, :]
            - (varg2 + varg2.T) / 2
        )
        + Xi4(**data) * (varom - varom.T)
        + Xi5(**data) * (
            vara.T
            - np.diag(vara)[None, :]
            + mu(**data) * (vara.T - vara)
        )
    )
    return LAMB

##############################################################
########## representation ####################################

def pairplot(Lamb,ax=None,title='',**kwargs):
    """Plot all the pairwize fitness (Lambda_ij,lambda_ji)
    of a Matrix  Lamb=(lambda_ij)_{ij}"""
    import matplotlib.pyplot as plt
    if ax is None : fig,ax=plt.subplots()
    n=len(Lamb)
    Lambhet=[]
    for i in range(n):
        for j in range(i+1,n):
            if j!=i: Lambhet.append([Lamb[i][j],Lamb[j][i]])
    Lambhet=np.array(Lambhet)
    ax.scatter(*Lambhet.T,**kwargs)
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    xl=max(np.abs(xlim[0]),np.abs(xlim[1]))
    yl=max(np.abs(ylim[0]),np.abs(ylim[1]))
    xlim=(-xl,xl)
    ylim=(-yl,yl)
    ax.plot(xlim,(0,0),'k')
    ax.plot((0,0),ylim,'k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title,fontsize=20)
