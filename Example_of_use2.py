# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:32:30 2026

@author: stenm
"""
import networkx as nx
import numpy as np
import Hosts_Replicator as hr

###################################
## Definition of the network ######
###################################

## Network from an adjacency matrix ##############

Ntot = 20  # number of nodes
p = 0.2    # probability of an edge between two nodes

# Generate a random graph
G = nx.erdos_renyi_graph(Ntot, p)

# Convert the graph to an adjacency matrix (NumPy array)
adj_matrix = nx.to_numpy_array(G, dtype=int)
###################################

###################################
### Number of strains #############
###################################
n = 10
######################################
## Variation from neutrality #######
######################################
## In this example there is a        #
## random variation in               #
## co-infection susceptibility       #
######################################
var_alpha = np.random.random((n, n)) - 0.5

######################################

data2 = {'CM': adj_matrix,
         'a': var_alpha,
         'sigma': 0.2}
Lamb = hr.lamb(**data2)
sol2 = hr.traject(T=1000, LAMBDA=Lamb)
hr.Final_figure(sol2, title='example 2')
