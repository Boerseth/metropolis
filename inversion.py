# -*- coding: utf-8 -*-
import time

import numpy as np
from matplotlib import pyplot as plt

from shakers import *
from metropolis import *

def trans3(G_h, G_v, G_dr, G_dl, N, L):
    I = np.identity(N)
    A = np.zeros((N,N))
    D = np.zeros((N,N))
    B = np.zeros((N,N))
    
    D += - np.diagflat( G_h[L-1], 1 )
    D += - np.diagflat( G_h[L-1], -1 )
    D += np.diagflat( G_h[L-1] + [0.0] )
    D += np.diagflat( [0.0] + G_h[L-1] )
    D += np.diagflat( G_v[L-2] )
    D += np.diagflat( G_dr[L-2] + [0.0] )
    D += np.diagflat( [0.0] + G_dl[L-2] )
    
    A += D
    
    for l in range(1, L-1):
        B  = - np.diagflat( G_v[L-1-l],   0 )
        B += - np.diagflat( G_dl[L-1-l],  1 )
        B += - np.diagflat( G_dr[L-1-l], -1 )
        
        D  = - np.diagflat( G_h[L-1-l], 1 )
        D += - np.diagflat( G_h[L-1-l], -1 )
        D += np.diagflat( G_h[L-1-l] + [0.0] )
        D += np.diagflat( [0.0] + G_h[L-1-l] )
        
        D += np.diagflat( G_v[L-1-l] )
        D += np.diagflat( G_v[L-2-l] )
        D += np.diagflat( G_dl[L-1-l] + [0.0] )
        D += np.diagflat( G_dr[L-2-l] + [0.0] )
        D += np.diagflat( [0.0] + G_dr[L-1-l] )
        D += np.diagflat( [0.0] + G_dl[L-2-l] )
        
        A = np.linalg.solve( D, B )
        A = D - np.transpose(B).dot(A)
        
    B  = - np.diagflat( G_v[0],   0 )
    B += - np.diagflat( G_dl[0],  1 )
    B += - np.diagflat( G_dr[0], -1 )
    
    D  = - np.diagflat( G_h[0], 1 )
    D += - np.diagflat( G_h[0], -1 )
    D += np.diagflat( G_h[0] + [0.0] )
    D += np.diagflat( [0.0] + G_h[0] )
    D += np.diagflat( G_v[0] )
    D += np.diagflat( G_dl[0] + [0.0] )
    D += np.diagflat( [0.0] + G_dr[0] )
    
    A = np.linalg.solve( D, B )
    A = D - np.transpose(B).dot(A)
    
    return A

def trans2(G_h, G_v, N, L):
    I = np.identity(N)
    A = np.zeros((N,N))
    S = np.zeros((N,N))
    U = np.zeros((N,N))
    
    S += - np.diagflat( G_h[L-1], 1 )
    S += - np.diagflat( G_h[L-1], -1 )
    S += np.diagflat( G_h[L-1] + [0.0] )
    S += np.diagflat( [0.0] + G_h[L-1] )
    
    A = S + np.transpose(np.linalg.solve( np.transpose( I + U.dot(A) ) , A ) )
    for l in range(1,L):
        u_temp = [ 1/g for g in G_v[L-1-l] ]
        U = np.diagflat(u_temp)
        
        S = np.zeros((N,N))
        S += - np.diagflat( G_h[L-1-l], 1 )
        S += - np.diagflat( G_h[L-1-l], -1 )
        S += np.diagflat( G_h[L-1-l] + [0.0] )
        S += np.diagflat( [0.0] + G_h[L-1-l] )
        
        A = S + A.dot( np.linalg.inv( I + U.dot(A) ) )
    return A


def trans(G_h, G_v,N,L):
    A_tra = np.zeros((N,N))
    g = 0.0
    for i in range(N-1):
        g = G_h[L-1][i]
        A_tra[i][i+1] += -g
        A_tra[i+1][i] += -g
        A_tra[i][i] += g
        A_tra[i+1][i+1] += g
        
    for l in range(1, L):
        # 1)
        for a in range(N):
            g = G_v[L-1-l][a]
            A = copy.copy(A_tra)
            for i in range(N):
                for j in range(N):
                    A_tra[i][j] = A[i][j] - A[i][a]*A[a][j]/(g + A[a][a])
        # 2)
        for i in range(N-1):
            g = G_h[L-1-l][i]
            A_tra[i][i+1] += -g
            A_tra[i+1][i] += -g
            A_tra[i][i] += g
            A_tra[i+1][i+1] += g
    return A_tra


"""
   Functions for turning a vector of numbers into a grid.
   
   The two lists-of-lists G_H and G_V contain resistor conductances in
   horizontal and vertical directions.
"""

def g_split(G, N, L):
    G_H = []
    G_V = []
    for l in range(L):  # R1 + R2  =  g1*g2 / ( g1 + g2 )
        G_H.append( G[l*(N-1) : l*(N-1) + N-1] )
    for l in range(L-1):
        G_V.append( G[L*(N-1) + l*N : L*(N-1) + l*N + N] )
    return G_H, G_V


def norm(A, G):
    N = len(A)
    L = (len(G) + N)/(2*N-1)
    
    GH, GV = g_split(G, N, L)
    
    return (
            np.linalg.norm( np.linalg.inv(A[:N-1,:N-1])
                            - np.linalg.inv(trans2(GH, GV, N, L)[:N-1,:N-1]) )
        +   np.linalg.norm(A - trans2(GH, GV, N, L) )
        )
    

def g_split2(G, N, L):
    G_H = []
    G_V = []
    for l in range(L-1):  # R1 + R2  =  g1*g2 / ( g1 + g2 )
        G_H.append( [ G[N*l+n]*G[N*l+n+1]/(G[N*l+n]+G[N*l+n+1]) for n in range(N-1)] )
        G_V.append( [ G[N*l+n]*G[N*(l+1)+n]/(G[N*l+n]+G[N*(l+1)+n]) for n in range(N)] )
    G_H.append( [ G[N*L-N+n]*G[N*L-N+n+1]/(G[N*L-N+n]+G[N*L-N+n+1]) for n in range(N-1)] )
    return G_H, G_V

def norm2(A, G):
    N = len(A)
    L = len(G)/N
    
    GH, GV = g_split2(G, N, L)
    
    return np.linalg.norm( A - trans(GH, GV, N, L) )**2
    
    
def g_split3(G, N, L):
    G_H = []
    G_V = []
    G_DR = []
    G_DL = []
    for l in range(L-1):  # R1 + R2  =  g1*g2 / ( g1 + g2 )
        G_H.append( [ G[N*l+n]*G[N*l+n+1]/(G[N*l+n]+G[N*l+n+1]) for n in range(N-1)] )
        G_DR.append([ np.sqrt(0.5)*G[N*l+n]*G[N*(l+1)+n+1]/(G[N*l+n]+G[N*(l+1)+n+1])
                        for n in range(N-1)] )
        G_DL.append([ np.sqrt(0.5)*G[N*(l+1)+n]*G[N*l+n+1]/(G[N*(l+1)+n]+G[N*l+n+1])
                        for n in range(N-1)] )
        G_V.append( [ G[N*l+n]*G[N*(l+1)+n]/(G[N*l+n]+G[N*(l+1)+n]) for n in range(N)] )
    G_H.append( [ G[N*L-N+n]*G[N*L-N+n+1]/(G[N*L-N+n]+G[N*L-N+n+1]) for n in range(N-1)] )
    return G_H, G_V, G_DR, G_DL
    
    
def norm3(A, G):
    N = len(A)
    L = len(G)/N
    
    GH, GV, GDR, GDL = g_split3(G, N, L)
    
    return np.linalg.norm( A - trans3(GH, GV, GDR, GDL, N, L) )**2

    
def pythagoras(x):
    return x[0]**2 + x[1]**2 + x[2]**2
    
def test_main():
    xl = -1.0
    xh = 1.0
    x = [1.0,1.0,1.0]
    T = 1.0
    print metropolis(pythagoras, x, xl, xh, shake_simple)
    
    
def find_g_from_A(A, N, L, g_min, g_max):
    f = lambda x: norm(A, x)
    g = [g_min]*( N*(L-1) + L*(N-1) )
    g = metropolis(f,g,g_min,g_max,shake_simple)
    return g
    
    
def test_test():
    A = np.array( [[1,-1,0],[-1,2,-1],[0,-1,1]] )
    g = find_g_from_A(A, 3, 1, 0.5, 1.5)
    gh, gv = g_split(g, 3, 1)
    A_est = trans(gh, gv, 3, 1)
    print A_est
    print A
    print g
    print np.linalg.norm( A - A_est )
    


def opposite(f, x, x_min, x_max, T_max):
    E = 0.0
    E_mean = []
    E_var = []
    
    T_list = np.linspace(0.0, T_max,50)
    
    for T in T_list:
        E, x, E_list = shake_and_record_simple2(f,x,x_min,x_max,T, 10)
        E_mean.append( sum(E_list)/len(E_list) )
        E_list = [ E_i**2 for E_i in E_list ]
        E_var.append( sum(E_list)/len(E_list) - E_mean[-1]**2 )
    return E_mean, E_var, T_list
    
def only_shuffle(f, x, x_min, x_max, T):
    E = f(x)
    E, x, E_list = shake_and_record_simple(f,x,x_min,x_max,T, 100)
    return E_list
        
    

def graph_of_E_mean_to_T(T_max):
    N = 12
    L = 3
    gmin = 1.0
    gmax = 2.0
    g = [ 0.5*(gmin+gmax) ]*( 2*N*L - N - L )
    gh, gv = g_split(g, N, L)
    A = trans(gh, gv, N, L)
    f = lambda x: norm(A, x)
    E_mean, E_var, T_list = opposite(f,g,gmin,gmax,T_max)
    plt.plot(T_list, E_mean)
    plt.plot(T_list, E_var)
    plt.show()
    
def histogram_of_E_at_T(T):
    N = 12
    L = 3
    gmin = 1.0
    gmax = 2.0
    g = [ 0.5*(gmin+gmax) ]*( 2*N*L - N - L )
    gh, gv = g_split(g, N, L)
    A = trans(gh, gv, N, L)
    f = lambda x: norm(A, x)
    E_list = only_shuffle(f, g, gmin, gmax, T)
    plt.hist(E_list, bins=30)
    
def new_test():
    N = 3
    L = 3
    g = [1,1,  1,1,1,  1,1,  1,1,1,  1,1,]
    gh, gv = g_split(g, N, L)
    print trans2(gh, gv, N, L)
    
def testing_testing():
    N = 3
    L = 2
    g = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    gh, gv, gdr, gdl = g_split3(g, N, L)
    print trans3(gh, gv, gdr, gdl, N, L)
    
def metro_rec():
    N = 12
    L = 3
    gmin = 1.0
    gmax = 2.0
    g = [ 0.5*(gmin+gmax) ]*( 2*N*L - N - L )
    gh, gv = g_split(g, N, L)
    A = trans(gh, gv, N, L)
    f = lambda x: norm(A, x)
    g = [gmin]*( 2*N*L - N - L )
    g_est, E_list, T_list = metropolis_record(f,g,gmin,gmax)
    plt.figure(figsize=(12,7))
    plt.plot(T_list, E_list)
    plt.show()
    plt.figure(figsize=(12,7))
    plt.plot(np.log(T_list), np.log(E_list))
    plt.show()
    print g_est
    
def metro_test():
    N = 12
    L = 3
    M = 5
    gmin = 1.0
    gmax = 2.0
    g = [ 0.5*(gmin+gmax) ]*( 2*N*L - N - L )
    gh, gv = g_split(g, N, L)
    A = trans(gh, gv, N, L)
    f = lambda x: norm(A, x)
    
    g_in = [ list(np.random.uniform(gmin, gmax, 2*N*L - N - L)) for i in range(M)]
    g_est = metropolis_with_avg(f,g_in,gmin,gmax,50)
    print g_est
    
    
def temper():
    separate = True
    start = time.time()
    N = 12
    L = 3
    gmin = 1.0
    gmax = 2.0
    
    M = 3
    if separate == True:
        g = [ 0.5*(gmin+gmax) ]*( 2*N*L - N - L )
        gh, gv = g_split(g, N, L)
        A = trans(gh, gv, N, L)
        f = lambda x: norm(A, x)
        g_in = [ [gmin]*( 2*N*L - N - L ) ]*M
    else:
        g = [ 0.5*(gmin+gmax) ]*(N*L)
        gh, gv = g_split2(g, N, L)
        A = trans(gh, gv, N, L)
        f = lambda x: norm2(A, x)
        g_in = [ [gmin]*(N*L) ]*M
    
    g_out = tempering(f,g_in,gmin,gmax,shake_simple)
    
    if separate == True:
        g_outH, g_outV = g_split(g_out[0], N, L)
        g_show = []
        g_show.append(g_outH[0] + [0.0])
        for l in range(L-1):
            g_show.append(g_outV[l])
            g_show.append(g_outH[l+1] + [0.0])
        print g_out[0]
        plt.imshow(g_show, interpolation="nearest")
    else:
        print g_out[0]
        g_show = list(np.reshape(g_out[0], (L,N)))
        plt.imshow(g_show, interpolation="nearest")
        
    stop = time.time()
    print stop - start
    
def temper_rec():
    separate = True
    start = time.time()
    N = 12
    L = 2
    gmin = 1.0
    gmax = 2.0
    
    M = 10
    if separate == True:
        g = [ 0.5*(gmin+gmax) ]*( 2*N*L - N - L )
        gh, gv = g_split(g, N, L)
        A = trans(gh, gv, N, L)
        f = lambda x: norm(A, x)
        g_in = []
        for i in range(M):
            g_in.append( list(np.random.uniform(gmin, gmax, 2*N*L - N - L)) )
    else:
        g = [ 0.5*(gmin+gmax) ]*(N*L)
        gh, gv = g_split2(g, N, L)
        A = trans(gh, gv, N, L)
        f = lambda x: norm2(A, x)
        g_in = [ [gmin]*(N*L) ]*M
    
    g_out, E_list, T_list = tempering_record(f,g_in,gmin,gmax)
    
    plt.figure(figsize=(12,7))
    plt.plot(T_list, E_list)
    plt.show()
    plt.figure(figsize=(12,7))
    plt.plot(np.log(T_list), np.log(E_list))
    plt.show()
    
    if separate == True:
        g_outH, g_outV = g_split(g_out[0], N, L)
        g_show = []
        g_show.append(g_outH[0] + [gmin])
        for l in range(L-1):
            g_show.append(g_outV[l])
            g_show.append(g_outH[l+1] + [gmax])
        print g_out[0]
        plt.imshow(g_show, interpolation="nearest")
    else:
        print g_out[0]
        g_show = list(np.reshape(g_out[0], (L,N)))
        g_show.append( [gmin]*(N-1) + [gmax] )
        plt.imshow(g_show, interpolation="nearest")
        
    stop = time.time()
    print stop - start

#new_test()    
#test_test()

#plt.figure(figsize=(12,7))
#graph_of_E_mean_to_T(4.0)
#plt.figure(figsize=(12,7))
#graph_of_E_mean_to_T(1.0)
#
#plt.figure(figsize=(12,7))
#histogram_of_E_at_T(2.0)
#plt.show()
#
#plt.figure(figsize=(12,7))
#histogram_of_E_at_T(2.0)
#histogram_of_E_at_T(0.3)
#histogram_of_E_at_T(0.1)
#histogram_of_E_at_T(0.03)
#plt.show()
#
metro_rec()
#temper()
#temper_rec()

#metro_test()