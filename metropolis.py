# -*- coding: utf-8 -*-
import copy

import numpy as np
import matplotlib.pyplot as plt

from shakers import shake_and_record_simple, shake_and_record_simple2, shake_simple

"""
    A *very* simple metropolis algorithm
    
    This is the metropolis algorithm in its simplest form; To find a minimum
    of the function  f  with starting point  x  :
      i)   randomly alter  x  repeatedly ("shake" the system) in a way that is 
           related to the temperature
      ii)  lower the temperature
    The "shakeing" is done by an outer function, which must be given as an
    argument.
"""
def metropolis(f, x, x_min, x_max, shake):
    E = f(x)
    T = E
    #while E < epsilon:
    for i in range(100):
        E, x = shake(f,x,x_min,x_max, T, 100)
        T = T*0.9
    return x
    
def metropolis_record(f, x, x_min, x_max):
    # Just get some value for T, and starting point...
    E0 = f(x)
    E1 = E0
    E = 0
    
    for i in range(100):
        u = list(np.random.uniform(x_min, x_max, len(x)))
        E1 = f(u)
        E += E1
        if E1 < E0:
            x = copy.copy(u)
            E0 = copy.copy(E1)
    # Over now...
    
    #T = E/100.0
    T = 1.0
    E = E0
    
    E_list = [E]
    T_list = [T]
    #while E < epsilon:
    for i in range(200):
        E, x, E_l = shake_and_record_simple2(f,x,x_min,x_max, T, 50)
        E_list.append(np.average(E_l))
        T_list.append(T)
        T = T*0.95
    return x, E_list, T_list


def metropolis_with_prob(f, x0, x_min, x_max, alpha):
    M = len(x0)
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = E0
    
    T = E0
    
    tries = [0]*M
    succs = [0]*M
    probs = [1.0]*M
    p_cum = [ sum(probs[:m+1]) for m in range(M)]
    
    for i in range(200):
        for j in range(alpha*M):
            """ Find index """
            stop = p_cum[-1]*np.random.random()
            index = 0
            while p_cum[index] < stop: #Very slow
                index = index + 1
                
            """ Make suggestion """
            succs[index] += 1 #Lower by 1 if unsuccessful
            tries[index] += 1
            x1[index] = x_min + (x_max-x_min)*np.random.random()
            E1 = f(x1)
            if np.random.random() < np.exp(-(E1-E0)/T):
                x0[index] = x1[index]
                E0 = E1
            else:
                x1[index] = x0[index]
                succs[index] += -1
        if i == 100:
            print tries
            print succs
            print
        if i == 199:
            print tries
            print succs
            print
        probs = [ max([ 0.01, float(s)/t ]) for (s,t) in zip(succs,tries) ]
        tries = [1]*M
        succs = [0]*M
        p_cum = [ sum(probs[:m+1]) for m in range(M)]
        T = T*0.9
    print probs
    return x0

def metropolis_with_prob2(f, x0, x_min, x_max, alpha):
    M = len(x0)
    u1 = copy.copy(x0)
    u2 = copy.copy(x0)
    u3 = copy.copy(x0)
    u4 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = E0
    E2 = E0
    E3 = E0
    E4 = E0
    
    T = E0
    
    tries = [1]*M
    succs = [1]*M
    probs = [1.0]*M
    p_cum = [ sum(probs[:m+1]) for m in range(M)]
    
    for i in range(200):
        for j in range(alpha*M):
            """ Find index """
            stop = p_cum[-1]*np.random.random()
            index = 0
            while p_cum[index] < stop: #Very slow
                index = index + 1
                
            """ Make suggestion """
            succs[index] += 1 #Lower by 1 if unsuccessful
            tries[index] += 1
            beta = 0.25*np.random.random()
            u1[index] = x_min + (x_max-x_min)*(beta)
            u2[index] = x_min + (x_max-x_min)*(beta + 0.25)
            u3[index] = x_min + (x_max-x_min)*(beta + 0.50)
            u4[index] = x_min + (x_max-x_min)*(beta + 0.75)
            E1 = f(u1)
            E2 = f(u2)
            E3 = f(u3)
            E4 = f(u4)
            if min([E1,E2,E3,E4]) < E0:
                if ( E1 < E2 and E1 < E3 ) and E1 < E4:
                    x0[index] = u1[index]
                    E0 = E1
                elif E2 < E3 and E2 < E4:
                    x0[index] = u2[index]
                    E0 = E2
                elif E3 < E4:
                    x0[index] = u3[index]
                    E0 = E3
                else:
                    x0[index] = u4[index]
                    E0 = E4
            else:
                ps = [1.0,
                      np.exp(-(E1-E0)/T),
                      np.exp(-(E2-E0)/T),
                      np.exp(-(E3-E0)/T),
                      np.exp(-(E4-E0)/T)]
                pscum = [ sum(ps[:m+1]) for m in range(4)]
                stop = pscum[-1]*np.random.random()
                if stop < pscum[0]:
                    succs[index] += -1
                elif stop < pscum[1]:
                    x0[index] = u1[index]
                    E0 = E1
                elif stop < pscum[2]:
                    x0[index] = u2[index]
                    E0 = E2
                elif stop < pscum[3]:
                    x0[index] = u3[index]
                    E0 = E3
                elif stop < pscum[4]:
                    x0[index] = u4[index]
                    E0 = E4
            u1[index] = x0[index]
            u2[index] = x0[index]
            u3[index] = x0[index]
            u4[index] = x0[index]
        if i == 100:
            print tries
            print succs
            print
        if i == 199:
            print tries
            print succs
            print
        #probs = [ max([ 0.01, float(s)/t ]) for (s,t) in zip(succs,tries) ]
        probs = [ float(s)/t for (s,t) in zip(succs,tries) ]
        #tries = [1]*M
        #succs = [0]*M
        p_cum = [ sum(probs[:m+1]) for m in range(M)]
        T = T*0.9
    print probs
    return x0

def metropolis_with_avg(f,x,xl,xh,n):
    M = len(x)
    E = [ f(x_i) for x_i in x ]
    E_avg0 = np.average(E)
    E_avg1 = E_avg0
    T = E_avg0
    E_list = [E_avg0]
    T_list = [T]
    
    #while E < epsilon:
    for i in range(10000):
        for m in range(M):
            E[m], x[m] = shake_simple(f,x[m],xl,xh, T, 1)
        E_avg1 = np.average(E)
        if E_avg1 < E_avg0:
            T = T*0.9
            E_avg0 = copy.copy(E_avg1)
        E_list.append(E_avg0)
        T_list.append(T)
    plt.plot(T_list, E_list)
    plt.show()
    plt.plot([np.log(T_i) for T_i in T_list], [np.log(E_i) for E_i in E_list])
    plt.show()
    return x[0]



def swap( E, x, T ):
    for m in range(len(E)-1):
        if np.random.uniform() < np.exp( (E[m+1] - E[m])*(1/T[m+1] - 1/T[m]) ):
            E_temp = E[m]
            x_temp = x[m]
            E[m] = E[m+1]
            x[m] = x[m+1]
            E[m+1] = E_temp
            x[m+1] = x_temp
    return E, x
    
def shake_temper(f, x0, xl, xh, T):
    M = len(x0)
    
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = copy.copy(E0)
    
    x1[np.random.randint(0,M)] = np.random.uniform(xl,xh)
    E1 = f(x1)
    if np.random.uniform(0.0,1.0) <= np.exp( - (E1-E0)/T ):
        x0 = copy.copy(x1)
        E0 = copy.copy(E1)
    else:
        x1 = copy.copy(x0)
            
    return E0, x0
    
def tempering(f, x, x_min, x_max, shake):
    M = len(x)
    E = [ f(x_m) for x_m in x]
    T = [ E[0]*1.1**k for k in range(M) ]
    
    for i in range(200):
        for j in range(M):
            for m in range(M):
                E[m], x[m] = shake_temper(f,x[m],x_min,x_max, T[m])
            E, x = swap( E, x, T )
        T = [T_m*0.9 for T_m in T]
    return x
    
def tempering_record(f, x, x_min, x_max):
    M = len(x)
    E = [ f(x_m) for x_m in x]
    E0 = min(E)
    T = [ E0*1.1**k for k in range(M) ]
    E_list = [E[0]]
    T_list = [T[0]]
    for i in range(200):
        E_l = []
        for j in range(5*M):
            for m in range(M):
                E[m], x[m] = shake_temper(f,x[m],x_min,x_max, T[m])
            E, x = swap( E, x, T )
            E_l.append(E[0])
        E_list.append( np.average(E_l) )
        T_list.append( T[0] )
        T = [T_m*0.9 for T_m in T]
    return x, E_list, T_list
    
    
    
    
    
    
    
    
    