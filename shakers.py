# -*- coding: utf-8 -*-

import copy as copy

import numpy as np


def shake_simple(f,x0,xl,xh,T,n):
    M = len(x0)
    
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = copy.copy(E0)
    
    for i in range(n*M):
        x1[np.random.randint(0,M)] = np.random.uniform(xl,xh)
        E1 = f(x1)
        if np.random.uniform(0.0,1.0) <= np.exp( - ((E1/T)**2 - (E0/T)**2) ):
            x0 = copy.copy(x1)
            E0 = copy.copy(E1)
        else:
            x1 = copy.copy(x0)
            
    return E0, x0



def shake_more(f,x0,xl,xh,T, n):
    M = len(x0)
    m = 4
    
    E0 = f(x0)
    
    for i in range(n):
        u = []
        F = []
        p = []
        
        for i in range(m):
            x1 = copy.copy(x0)
            x1[np.random.randint(0,M)] = np.random.uniform(xl,xh)
            E1 = f(x1)
            u.append( x1 )
            F.append( E1 )
            p.append( np.exp(-(E1-E0)/T) )
        
        F_min, j = min((val, idx) for (idx, val) in enumerate(F))
        if F_min < E0:
            x0 = u[j]
            E0 = F[j]
        else:
            S = sum(p)
            p_cum = 0.0
            
            xi = S*np.random.uniform(0.0,1.0)
            
            j = 0
            p_cum += p[j]
            while p_cum <= xi and j<len(p) - 1:
                j += 1
                p_cum += p[j]
            x0 = u[j]
            E0 = F[j]
    
    return E0, x0
    
    

def shake_and_record_simple(f,x0,xl,xh,T,alpha):
    M = len(x0)
    
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = copy.copy(E0)
    E_list = []
    E_list.append(E0)
    
    swap_count = 0
    while swap_count < alpha*M:
        x1[np.random.randint(0,M)] = np.random.uniform(xl,xh)
        E1 = f(x1)
        if np.random.uniform(0.0,1.0) <= np.exp( - ((E1/T)**2 - (E0/T)**2) ):
            x0 = copy.copy(x1)
            E0 = copy.copy(E1)
            E_list.append(E0)
            swap_count += 1
        else:
            x1 = copy.copy(x0)
                
    return E0, x0, E_list
    

def shake_and_record_simple2(f,x0,xl,xh,T,alpha):
    M = len(x0)
    
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = copy.copy(E0)
    E_list = []
    E_list.append(E0)
    
    for i in range(alpha*M):
        x1[np.random.randint(0,M)] = np.random.uniform(xl,xh)
        E1 = f(x1)
        if np.random.uniform(0.0,1.0) <= np.exp( - (E1-E0)/T ):
            x0 = copy.copy(x1)
            E0 = copy.copy(E1)
            E_list.append(E0)
        else:
            x1 = copy.copy(x0)
                
    return E0, x0, E_list