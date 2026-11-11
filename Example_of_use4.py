# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:20:00 2026

@author: stenm
"""

import numpy as np
import matplotlib.pyplot as plt
import Hosts_Replicator as hr

######## Example 3 ##################
#####################################
## Stabilization by heterogeneity  ##
#####################################
## Compare two networks with the   ##
## same mean degree (kbar = 6)     ##
## for the exact same 8 strains:   ##
## - Homogeneous -> limit cycle    ##
## - Heterogeneous -> equilibrium  ##
#####################################

I_want_random = False # Set to True to explore new random matrices var_a
seed0 = 87716        # Seed giving a clear transition from cycle to equilibrium

if I_want_random:
    seed_use = np.random.randint(0, 100000)
    print(f"Generated seed: {seed_use}")
else:
    seed_use = seed0

#####################################
### Number of strains & variation ###
#####################################
rng = np.random.default_rng(seed_use)
n = 8
var_a = rng.random((n, n)) - 0.5

X0 = None

#####################################
### Networks (same mean kbar = 6) ###
#####################################
networks = [
    ([6], [1.0], 'Homogeneous network (large mu -> sustained cycle)'),
    ([1,30], [24/29, 5/29], 'Heterogeneous network (small mu -> stabilized)')
]

#####################################
### Computation and figures #########
#####################################
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (kappa, P, title) in zip(axes, networks):
    Lamb = hr.lamb(Kappa=kappa, P=P, beta=0.25, r=0.5, gamma=0.5, sigma=0.5, a=var_a)
    sol = hr.traject(T=10000, LAMBDA=Lamb, X0=X0)
    hr.Final_figure(sol, ax=ax, title=title, lw=2.5)

plt.tight_layout()
plt.show()