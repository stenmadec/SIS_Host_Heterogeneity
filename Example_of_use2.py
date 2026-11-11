# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:32:30 2026

@author: stenm
"""
import networkx as nx
import numpy as np
import Hosts_Replicator as hr
########Example 2##################
###################################
##Definition of the network #######
###################################

##Network with the adgency matrix###################

Ntot = 20  # nombre de sommets
p = 0.2  # probabilité d'avoir une arête entre deux nœuds

# Génère un graphe aléatoire
G = nx.erdos_renyi_graph(Ntot, p)

# Convertit le graphe en matrice d'adjacence (sous forme de tableau NumPy)
adj_matrix = nx.to_numpy_array(G, dtype=int)
###################################

###################################
### number of strains #############
###################################
n=10
######################################
## Variation from theterogeneity #####
######################################
## In this example there is a        #
## random variation in the           #
## co-infection                      #
######################################
var_alpha=np.random.random((n,n))-0.5

######################################

data2={'CM':adj_matrix,
      'a':var_alpha,
      'sigma':0.2}
Lamb=hr.lamb(**data2)
sol2=hr.traject(T=1000,LAMBDA=Lamb)
hr.Final_figure(sol2,title='example 2')


