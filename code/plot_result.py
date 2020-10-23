# !python3
# coding: utf-8
# author: Ziqi Liu and Zhi-Qin John Xu
# Reference: Ziqi Liu，Wei Cai，Zhi-Qin John Xu. Multi-scale Deep Neural Network (MscaleDNN)
# for Solving Poisson-Boltzmann Equation in Complex Domains[J]. 2020. arXiv:2007.11207 (CiCP)

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import matplotlib
matplotlib.use('Agg')      
Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88 - Leftp
Heightp = 0.9 - Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]
 
def save_fig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0):# Save the figure
    if isax==1:
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm = '%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm = '%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic==1:
        pltm.show() 
    elif isShowPic==-1:
        return
    else:
        pltm.close()
        

with open(os.path.join("..", "exp", "sin_N", "80509", "data.pkl"), "rb") as f:
    R1 = pickle.load(f)
with open(os.path.join("..", "exp", "sin_M", "62178", "data.pkl"), "rb") as f:
    R2 = pickle.load(f) 

            
record_path=os.path.join("..", "exp")        
plt.figure()
ax = plt.gca()
plt.semilogy(R1["error"], label="Normal")
plt.semilogy(R2["error"], label="Mscale") 
plt.legend(fontsize=14)
fntmp = os.path.join(record_path, 'error') 
save_fig(plt,fntmp,ax=ax,isax=1,iseps=0) 
    
for k in [0, 500, 1000, 5000]:
    plt.figure()
    plt.plot(R2["x_samp"], R2["u_samp_"+str(k)], 'r--',label='dnn')
    plt.plot(R2["x_samp"], R2["u_samp_true"], 'b-',label='true')
    plt.legend(fontsize=14)
    plt.title('epoch '+str(k),fontsize=14)
    fntmp = os.path.join(record_path, 'u'+str(k)) 
    save_fig(plt,fntmp,ax=ax,isax=1,iseps=0) 
