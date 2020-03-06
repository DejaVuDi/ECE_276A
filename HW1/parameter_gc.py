"""
Get Gaussian Model parameters for BDR

@author Di Gu
"""

import numpy as np

colors = ['blue','brown','gray','green','red','black','tree','lightblue']
priors = np.zeros([len(colors)])
i = 0

def Gussian(color,i):
    c = np.load(color+'.npy')
    priors[i] = c.shape[0]
    mu = np.mean(c,0) 
    sigma = np.cov(c.T)
    print(color,mu,sigma)
    
G = [Gussian(colors[i],i) for i in range(len(colors))]
priors = priors/sum(priors)
print(priors)