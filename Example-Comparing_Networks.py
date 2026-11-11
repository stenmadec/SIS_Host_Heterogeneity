# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:18:51 2026

@author: stenm
"""


import numpy as np
import Hosts_Replicator as hr
import matplotlib.pyplot as plt
import matplotlib
##Generation of the exact figure or generation of a new one
networkisfixe=True
varstrainisfixe=True
##standard deviation of a pairplot
def stdnet(mom2,kbar):
    return np.sqrt(np.max((mom2-kbar**2,0)))

###Generation of the perturbations
matplotlib.rcParams['axes.labelsize'] = 20
######### Epidemiological parameters
data={'sigma':0.3,
      'beta':0.15,
      'kbar':20,
      'r':0.5,
      'gamma':0.5}


n=5# Just if we need
if varstrainisfixe:


    ###example for the figure 4 of the paper 
    ##How host contact network impacts $N$-strain SIS dynamics with coinfection via a global replicator equation

    vg=np.array([[-0.36272215,  0.13257483, -0.28100324, -0.00625503, -0.09398224,
             0.06513616,  0.19350629,  0.01637501,  0.01412913,  0.18524046],
           [ 0.11182475,  0.08100528, -0.09656951,  0.46808387,  0.09399792,
             0.39634958, -0.27325708,  0.28347766, -0.2171042 ,  0.28588191],
           [-0.07947499, -0.21073295, -0.35758215, -0.22147952, -0.15502074,
             0.14733105, -0.2943999 ,  0.39052718,  0.13503741,  0.22720563],
           [ 0.34782619,  0.13148508,  0.06019466, -0.39855906,  0.41617588,
            -0.34960678,  0.14336247,  0.1995688 , -0.30864808,  0.48118337],
           [-0.45415165, -0.46734788,  0.34338363, -0.18462672, -0.30237686,
            -0.16848815, -0.1641655 ,  0.24692417,  0.47685097,  0.36548444],
           [ 0.05618164,  0.04502053,  0.47353285,  0.11887153,  0.47111098,
            -0.19865101, -0.41190156,  0.26417511,  0.04392377,  0.22383209],
           [-0.00898671, -0.29336341, -0.14470324, -0.15249758,  0.08818061,
            -0.02145757, -0.30425227, -0.22484969, -0.32874807, -0.45231762],
           [-0.25791188,  0.38146971, -0.13052268, -0.28806907, -0.36614744,
             0.00703833, -0.21283422,  0.09374675, -0.2188067 ,  0.42231233],
           [-0.16870523,  0.41840743, -0.43697821,  0.20572794, -0.10864175,
             0.21840758,  0.15035724, -0.06321128,  0.00425692, -0.07577719],
           [ 0.39237529, -0.46840839, -0.31256299,  0.06572951,  0.03055605,
             0.26492367, -0.48807796,  0.31170529,  0.05068493,  0.35946858]])
    
    n=len(vg)
else : 
    vg=np.random.random((n,n))-0.5
data['a']=vg
##generation of the Network
numbernet=1
kappa0,P0,mom0=[data['kbar']],[1.],data['kbar']**2
if networkisfixe:
    #kappa1,P1,mom1=Gen_Distrib(kbar,num=numbernet,distrib=Distribution_three)[0]
    kappa1=[10, 20, 30]
    P1=[0.4, 0.20, 0.4]
    mom1=sum([pk*k*k for pk,k in zip(P1,kappa1)])
    kappa2=[k+1 for k in range(97)]
    P2=np.array([6.76446264e-03, 1.60062889e-02, 5.75301100e-02, 5.80575576e-02,
           1.81985619e-02, 3.28085689e-02, 1.87481515e-02, 5.01486916e-02,
           2.27670719e-02, 1.95029749e-02, 1.35902369e-02, 5.83350137e-02,
           4.33938713e-02, 5.74351118e-02, 6.69260396e-02, 6.38659871e-02,
           4.66774388e-02, 8.96168251e-02, 8.28943102e-02, 4.01423928e-03,
           8.85433248e-04, 1.22534954e-04, 3.21034245e-03, 2.86578928e-03,
           1.61441821e-03, 3.59613839e-03, 1.71451789e-03, 1.27154035e-03,
           2.54791985e-03, 1.73788550e-03, 1.99003259e-03, 2.23508982e-03,
           2.58104671e-03, 1.19263746e-04, 1.16721756e-04, 3.32389087e-03,
           1.58017539e-03, 1.54401951e-04, 5.47710684e-04, 4.38796112e-03,
           4.27377241e-03, 2.61037856e-03, 3.13418998e-03, 1.33399146e-03,
           3.87625774e-03, 3.15269337e-03, 3.61809183e-03, 3.90319195e-03,
           1.62966987e-03, 3.97257227e-03, 8.98625141e-05, 3.04452107e-03,
           2.88108854e-03, 3.85783269e-03, 4.46072386e-03, 6.99579754e-04,
           3.55171088e-03, 1.89666686e-03, 1.70492400e-03, 2.80381120e-03,
           2.72975554e-03, 3.42642004e-03, 1.43161672e-03, 3.60684628e-03,
           3.17262876e-03, 2.16593850e-03, 3.70813918e-03, 2.32047614e-03,
           3.84956051e-03, 2.02150017e-03, 6.34580966e-04, 4.55327197e-03,
           6.67814957e-04, 3.68295373e-03, 6.35770267e-04, 3.70477073e-03,
           1.23467121e-03, 3.02779289e-03, 3.86304973e-04, 2.27874464e-03,
           1.49972564e-03, 1.99673459e-03, 4.10254446e-03, 4.13097432e-03,
           1.04421178e-03, 1.20772922e-03, 2.46824931e-03, 2.19840565e-04,
           1.98012436e-03, 3.12549025e-03, 1.50982093e-03, 1.36208972e-03,
           4.00023700e-04, 2.53680728e-03, 9.10473733e-04, 3.68324307e-04,
           1.51741853e-03])
    mom2= sum([pk*k*k for pk,k in zip(P2,kappa2)])
    ListNet=[(kappa0,P0,mom0),(kappa1,P1,mom1),(kappa2,P2,mom2)]
    max_kappa=40.
else:
    ListNet=hr.Gen_Distrib(kbar=data['kbar'],num=numbernet,distrib='random')
    ListNet.append((kappa0,P0,mom0))
    ListNet+=hr.Gen_Distrib(kbar=data['kbar'],num=numbernet,distrib='three')
    max_kappa = max(max(net[0]) for net in ListNet)
Titre_ex=['Random network\n','Homogeneous Network\n','3-class network\n']
ListNet =[(*net, titre) for  net,titre in zip(ListNet,Titre_ex)]
ListNet.sort(key=lambda x:stdnet(x[2],data['kbar']))
sig=[stdnet(x[2],data['kbar']) for x in ListNet]


######Generation des replicateurs et trajectoires
listLambda,listTitle,listres=[],[],[]

for x,s in zip(ListNet,sig):
    data['Kappa']=x[0]
    data['P']=x[1]
    title=x[3]
    res,LL=hr.Final_Computation(n=n,T=1000,**data)
    listLambda.append(LL)
    title+=r'$\text{std}\left(\mathcal{N}\right)=$'+f'{s:.1f}'
    listTitle.append(title)
    listres.append(res)
##3)Generation of the plots
Nplot=len(sig)
fig,AX=plt.subplots(ncols=Nplot,nrows=3,figsize=(16,16))
plt.subplots_adjust(wspace=0.4)
for LL,listnet,ax,title,res,cas in zip(listLambda,ListNet, AX.T,
                                       listTitle,
                                       listres,
                                       [1,2,3]):
    kappa,P,_,_=listnet
    hr.bardistrib(kappa,P,ax=ax[0],title=title)
    ax[0].set_title(title,fontsize=30)
    ax[0].set_xlim((0,max_kappa ))
    ax[0].set_xlabel('degree distribution k')
    ax[1].set_xlim((-1,1))
    ax[1].set_ylim((-1,1))
    lambdat=[-1.,-0.5,0.,0.5,1.]
    ax[1].set_yticks(lambdat)
    ax[1].set_yticklabels([str(ti) for ti in lambdat],fontsize=20)
    ax[1].set_xticks(lambdat)
    ax[1].set_xticklabels([str(ti) for ti in lambdat],fontsize=20)
 
    if cas ==1 :
        ax[0].set_ylabel( 'Prob. distribution, '+r' $p_k$',
                         fontsize=30)
        ax[1].set_ylabel( r'$(\lambda_i^j,\lambda_j^i)$',fontsize=30)
        ax[2].set_ylabel(r'Dynamics $\mathbf{z}(\tau)$',
                         fontsize=30)
        
    hr.pairplot(LL,ax=ax[1],title='',alpha=0.5,s=100,c='k')
    
    kappat=np.linspace(0,max_kappa,5)
    pkt=[k/5 for k in range(6)]
    ax[0].set_yticks(pkt)
    ax[0].set_yticklabels([f'{ti:0.2f}' for ti in pkt],fontsize=20)
    if cas==3 : 
        pkt=[k/50 for k in range(6)]
        ax[0].set_yticks(pkt)
        ax[0].set_yticklabels([f'{ti:0.2f}' for ti in pkt],fontsize=20)
        ax[0].set_ylim([0,0.1])
    ax[0].set_xticks(kappat)
    ax[0].set_xticklabels([str(ti) for ti in kappat],fontsize=20)

    hr.Final_figure(res,ax=ax[2],title='',lw=4)
    timet=[100*k for k in [0,2,4,6,8,10]]
    ax[2].set_xticks(timet)
    ax[2].set_xticklabels([str(ti) for ti in timet],fontsize=20)
    zt=[0.0,0.2,0.4,0.6,0.8,1.]
    ax[2].set_yticks(zt)
    ax[2].set_yticklabels([str(ti) for ti in zt],fontsize=20)
    ax[2].set_xlabel('Time')
    #put degree disrtibituin $k$ xlabel on first line
    #increase the spacing h
    