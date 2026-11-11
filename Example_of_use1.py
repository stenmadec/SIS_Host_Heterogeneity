# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:36:02 2026

@author: stenm
"""
import numpy as np
import Hosts_Replicator as hr


########Example 1##################
###################################
##Definition of the network #######
###################################
##Manual network###################
Kappa=[1,3,5,15]
P=[80/100,15/100,4/100 ,1/100]
###################################

###################################
### number of strains #############
###################################
n=5
######################################
## Variation from theterogeneity #####
######################################
## In this example there is a        #
## random variation in the           #
## clearance of the co-infected host #
######################################
var_gamma2=np.random.random((n,n))-0.5

######################################

data1={'Kappa':Kappa,
      'P':P,
      'c2':var_gamma2}
Lamb=hr.lamb(**data1)
sol1=hr.traject(T=1000,LAMBDA=Lamb)
hr.Final_figure(sol1)
plt.show()

