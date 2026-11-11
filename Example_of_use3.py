# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:32:00 2026

@author: stenm
"""

import numpy as np
import matplotlib.pyplot as plt
import Hosts_Replicator as hr

# 1. Define 4 strains with a trade-off between transmission (b) and priority (w)
np.random.seed(12)
n = 4
var_b = np.array([0.4, 0.2, -0.1, -0.3])   # strain 0 transmits faster
var_w = np.random.uniform(-0.8, 0.8, (n, n))

# 2. Compare two networks with the same mean degree (kbar=6) but different heterogeneity
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, mode, title in zip(axes, ['unif', 'random'], ['Quasi-homogeneous network', 'Heterogeneous network']):
    kappa, P, _ = hr.Gen_Distrib(kbar=6, num=1, distrib=mode)[0]
    
    Lamb = hr.lamb(Kappa=kappa, P=P, beta=0.4, sigma=0.5, b=var_b, w=var_w)
    sol = hr.traject(T=600, LAMBDA=Lamb)
    
    hr.Final_figure(sol, ax=ax, title=title, lw=2.5)

plt.tight_layout()
plt.show()