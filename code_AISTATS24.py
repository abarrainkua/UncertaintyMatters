# This program is free software: you can redistribute it and/or modify it under the terms of 
# the GNU General Public License as published by the Free Software Foundation, either version 
# 3 of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program. 
#If not, see <https://www.gnu.org/licenses/>.

# Uncertainty Matters

import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as  mpatches
from sklearn.neighbors import KernelDensity
import math
from sklearn.neighbors import KDTree

def hdr_Nd(X, prob = 90, extend = 0.15):
    """
    This function returns the (1-alpha)% HDR of a N-dimensional distribution.

    Args:
        X (array): An array containing the samples of the distribution. Each 
        prob (int) : The coverage of the HDR, i.e, prob% HDR.
        Default = 90.
        extend (float) : The bandwidth of the kernel in Kernel Density Estimation (KDE).
        Default =  0.15

    Returns:
        post (array): the samples within the (1-alpha)% HDR.
    """

    alpha = (100-prob)/100
    den = KernelDensity(kernel='gaussian', bandwidth=extend).fit(X)
    fxy = den.score_samples(X)
    falpha = np.quantile(fxy, alpha)
    points_Nd_hdr = X[fxy > falpha]

    return points_Nd_hdr

def posterior_method_2d(CMp, CMu, N = 10000, prior = [1,1,1,1], m1 = 'accuracy', m2 = 'dtpr'):
    """
    This function returns the 2-dimensional posterior distributions on (m1,m2)
    for a given method.

    Args:
        CMp (array): The confusion matrix of the privileged group, in the form [tn, fp, fn, tp].
        CMu (array): The confusion matrix of the unprivileged group, in the form [tn, fp, fn, tp].
        N (int) : Number of samples to approximate the posterior distribution.
        Default = 10000.
        prior (array): the prior for the multinomial parameter.
        m1 (string) : the first metric of the joint distribution.
        m2 (string) : the second metric of the joint distribution.

    Returns:
        post (array Nx2): the posterior distribution samples.
    """

    m1_s = np.zeros(N) 
    m2_s = np.zeros(N) 

    # Update the multinomial parameter:
    
    ap1 = [sum(x) for x in zip(CMp, prior)]
    au1 = [sum(x) for x in zip(CMu, prior)]

    Np = np.sum(CMp) # number of privileged instances
    Nu = np.sum(CMu) # number of unprivileged instances

    # Get the samples of the posterior distribution:

    for i in range(N):
        pi_p = np.random.dirichlet(ap1) # MTN parameter of privileged
        pi_u = np.random.dirichlet(au1) # MTN parameter of unprivileged
    
        pi_c = (pi_p * np.sum(CMp) + pi_u * np.sum(CMu)) / (np.sum(CMp) + np.sum(CMu)) # whole dataset

        CMi_p = np.random.multinomial(Np, pi_p)
    
        CMi_u = np.random.multinomial(Nu, pi_u)
    
        CMi = [sum(x) for x in zip(CMi_p, CMi_u)]


        if m1 == 'acc':
            m1_s[i] = (CMi[0] + CMi[3])/ np.sum(CMi)
        elif m1 == 'dtpr' :
            m1_s[i] = CMi_p[3] / (CMi_p[3] + CMi_p[2]) -  CMi_u[3] / (CMi_u[3] + CMi_u[2])
        elif m1 == 'dfpr' :
            m1_s[i] = CMi_p[1] / (CMi_p[1] + CMi_p[0]) -  CMi_u[1] / (CMi_u[1] + CMi_u[0])
        elif m1 == 'dp':
            m1_s[i] = (CMi_p[1] + CMi_p[3])/np.sum(CMi_p) - (CMi_u[1] + CMi_u[3])/np.sum(CMi_u)
        elif m1 == 'pp':
            m1_s[i] = CMi_p[3] / (CMi_p[1] + CMi_p[3]) - CMi_u[3] / (CMi_u[1] + CMi_u[3])

        if m2 == 'acc':
            m2_s[i] = (CMi[0] + CMi[3])/ np.sum(CMi)
        elif m2 == 'dtpr' :
            m2_s[i] = CMi_p[3] / (CMi_p[3] + CMi_p[2]) -  CMi_u[3] / (CMi_u[3] + CMi_u[2])
        elif m2 == 'dfpr' :
            m2_s[i] = CMi_p[1] / (CMi_p[1] + CMi_p[0]) -  CMi_u[1] / (CMi_u[1] + CMi_u[0])
        elif m2 == 'dp':
            m2_s[i] = (CMi_p[1] + CMi_p[3])/np.sum(CMi_p) - (CMi_u[1] + CMi_u[3])/np.sum(CMi_u)
        elif m2 == 'pp':
            m2_s[i] = CMi_p[3] / (CMi_p[1] + CMi_p[3]) - CMi_u[3] / (CMi_u[1] + CMi_u[3])


        post = np.array([[m1_s[i], m2_s[i]] for i in range(N)])


    return post

def posterior_group_2d(CM, N = 10000, prior = [1,1,1,1], m1 = 'tpr', m2 = 'fpr'):
    """
    This function returns the 2-dimensional posterior distributions on (m1,m2)
    for a given method.

    Args:
        CM (array): The confusion matrix of the group, in the form [tn, fp, fn, tp].
        N (int) : Number of samples to approximate the posterior distribution.
        Default = 10000.
        prior (array): the prior for the multinomial parameter.
        m1 (string) : the first metric of the joint distribution.
        m2 (string) : the second metric of the joint distribution.

    Returns:
        post (array Nx2): the posterior distribution samples.
    """

    m1_s = np.zeros(N) 
    m2_s = np.zeros(N) 

    # Update the multinomial parameter:
    
    ap = [sum(x) for x in zip(CM, prior)]

    Ni = np.sum(CM) # number of instances

    # Get the samples of the posterior distribution:

    for i in range(N):
        pi = np.random.dirichlet(ap) # MTN parameter

        CMi = np.random.multinomial(Ni, pi)

        if m1 == 'acc':
            m1_s[i] = (CMi[0] + CMi[3])/ np.sum(CMi)
        elif m1 == 'tpr' :
            m1_s[i] = CMi[3] / (CMi[3] + CMi[2])
        elif m1 == 'fpr' :
            m1_s[i] = CMi[1] / (CMi[1] + CMi[0])
        elif m1 == 'ar':
            m1_s[i] = (CMi[1] + CMi[3]) / np.sum(CMi)
        elif m1 == 'ppv':
            m1_s[i] = CMi[3] / (CMi[1] + CMi[3])


        if m2 == 'acc':
            m2_s[i] = (CMi[0] + CMi[3])/ np.sum(CMi)
        elif m2 == 'tpr' :
            m2_s[i] = CMi[3] / (CMi[3] + CMi[2])
        elif m2 == 'fpr' :
            m2_s[i] = CMi[1] / (CMi[1] + CMi[0])
        elif m2 == 'ar':
            m2_s[i] = (CMi[1] + CMi[3]) / np.sum(CMi)
        elif m2 == 'ppv':
            m2_s[i] = CMi[3] / (CMi[1] + CMi[3])


        post = np.array([[m1_s[i], m2_s[i]] for i in range(N)])


    return post



def comparison_2d(post1, post2):
    """
    This function returns the 2-dimensional posterior distributions of the performance
    difference between two methods.

    Args:
        post1 (array) : posterior distribution samples for method 1.
        post2 (array) : posterior distribution samples for method 2.

    Returns:
        diff (array Nx2): the posterior distribution samples of the difference.
    """

    return np.array([[(post1[i,0]-post2[i,0]),(post1[i,1]-post2[i,1])] for i in range(len(post1))])


def comparison_probabilities(diff, rope_m1 = 0.01, rope_m2 = 0.01):

    """
    This function returns the probabilities P(A>>B), P(B>>A), P(A=B).

    Args:
        diff (array) : the distribution of the performance difference.
        rope_m1 (float) : rope dimension for m1. Default = 0.01
        rope_m2 (float) : rope dimension for m2. Default = 0.01

    Returns:
        p (array): the probabilities P(A>>B), P(B>>A), P(A=B).
    """

    p_a = 0
    p_b = 0
    p_ab = 0

    for i in range(len(diff)):
        if (abs(diff[0,i]) < rope_m1 and  abs(diff[1,i]) < rope_m2 ):
            p_ab += 1

        else:
            if diff[0,i] > -rope_m1:
                if (-diff[1,i] > -rope_m2):
                    p_a += 1
                
            if diff[0,i] < rope_m1:
                if (-diff[1,i] < rope_m2):
                    p_b += 1    


    p = [p_a/len(diff_acc), p_b/len(diff_acc), p_ab/len(diff_acc)]

    return p


###################### EXAMPLE #########################

#CM1P = [ x * 0.25 for x in [466, 61, 130, 33]]
#CM1U = [ x * 0.25 for x in [185, 28, 81, 16]]

#CM2P = [ x * 0.711 for x in [537, 76, 144, 53]]
#CM2U = [ x * 0.711 for x in [101, 21, 59, 9]]

CM1P =  [466, 61, 130, 33]
CM1U = [185, 28, 81, 16]

CM2P = [537, 76, 144, 53]
CM2U = [101, 21, 59, 9]

metric1_comp = 'acc'
metric2_comp = 'dtpr'

metric1 = 'tpr'
metric2 = 'fpr'

rope_m1 = 0.01
rope_m2 = 0.01

fig, axs = plt.subplots(2, 2, figsize = (14,18))


# posterior distributions for the majority and the minority
# for method 1

post1_p = posterior_group_2d(CM1P, N = 10000, prior = [1,1,1,1], m1 = metric1, m2 = metric2)

print(post1_p)
post1_u = posterior_group_2d(CM1U, N = 10000, prior = [1,1,1,1], m1 = metric1, m2 = metric2)

sns.kdeplot(x=post1_u[:, 0], y=post1_u[:,1], cmap="BuGn", fill=True, bw_adjust=1, ax=axs[0,0], label = 'Minority')
sns.kdeplot(x=post1_p[:, 0], y=post1_p[:,1], cmap='RdPu', fill=True, bw_adjust=1, ax=axs[0,0], label = 'Majority')
axs[0,0].set_title('method1', fontsize=18)
axs[0,0].set_xlabel(metric1, fontsize = 15)
axs[0,0].set_ylabel(metric2, fontsize = 15)
handles = [mpatches.Patch(facecolor=plt.cm.RdPu(100), label="Majority"),
           mpatches.Patch(facecolor=plt.cm.BuGn(100), label="Minority")]
axs[0,0].legend(handles=handles, loc = 'upper left')

# posterior distributions for the majority and the minority
# for method 2

post2_p = posterior_group_2d(CM2P, N = 10000, prior = [1,1,1,1], m1 = metric1, m2 = metric2)
post2_u = posterior_group_2d(CM2U, N = 10000, prior = [1,1,1,1], m1 = metric1, m2 = metric2)

sns.kdeplot(x=post2_u[:,0], y=post2_u[:,1], cmap="BuGn", fill=True, bw_adjust=1, ax=axs[0,1], label = 'Minority')
sns.kdeplot(x=post2_p[:,0], y=post2_p[:,1], cmap='RdPu', fill=True, bw_adjust=1, ax=axs[0,1], label = 'Majority')
axs[0, 1].set_title('method2', fontsize=18)
axs[0,1].set_xlabel(metric1, fontsize = 15)
axs[0,1].set_ylabel(metric2, fontsize = 15)
handles = [mpatches.Patch(facecolor=plt.cm.RdPu(100), label="Majority"),
           mpatches.Patch(facecolor=plt.cm.BuGn(100), label="Minority")]
axs[0,1].legend(handles=handles, loc = 'upper left')

# Posterior distribution of method 1 and method 2

post1 = posterior_method_2d(CM1P, CM1U, N = 10000, prior = [1,1,1,1], m1 = metric1_comp, m2 = metric2_comp)
post2 = posterior_method_2d(CM2P, CM2U, N = 10000, prior = [1,1,1,1], m1 = metric1_comp, m2 = metric2_comp)

sns.kdeplot(x=post1[:,0], y=post1[:,1], cmap="Blues", fill=True, bw_adjust=1, ax =axs[1,0], label = 'method 1')
sns.kdeplot(x=post2[:,0], y=post2[:,1], cmap="Oranges", fill=True, bw_adjust=1, ax =axs[1,0], label = 'method 2')
axs[1,0].set_xlabel(metric1_comp, size = 18)
axs[1,0].set_ylabel(metric2_comp, size = 18)
handles = [mpatches.Patch(facecolor=plt.cm.Blues(100), label='method1'),
           mpatches.Patch(facecolor=plt.cm.Oranges(100), label='method2')]
axs[1,0].legend(handles=handles, loc = 'upper left')

# Posterior distribution of method 1 vs. method 2

diff = comparison_2d(post1, post2)
hdr_diff = hdr_Nd(diff, prob = 95, extend = 0.15)
print(diff)
sns.kdeplot(x=hdr_diff[:,0], y=-hdr_diff[:,1], cmap="Blues", fill=True, ax = axs[1,1], bw_adjust=1.5)
axs[1,1].set_xlabel('\u0394'+ 'Accuracy' , fontsize = 22)
axs[1,1].set_ylabel('\u0394'+ 'Fairness' , fontsize = 22)
axs[1,1].vlines(x = [-rope_m1, rope_m1], ymin = -1, ymax = 1, color = 'blue', ls = '--')
axs[1,1].hlines(y = [-rope_m2, rope_m2], xmin = -1, xmax = 1, color = 'blue', ls = '--')
axs[1,1].axhline(0.0, color = 'black', ls = '--')
axs[1,1].axvline(0.0, color = 'black', ls = '--')
axs[1,1].set_xlim(-0.2, 0.2)
axs[1,1].set_ylim(-0.4, 0.4)

plt.show()





