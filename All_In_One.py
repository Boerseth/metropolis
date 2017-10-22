# -*- coding: utf-8 -*-
import copy
import time

import numpy as np
from scipy import linalg
from scipy.stats import norm
import matplotlib.pyplot as plt

N = 100
L = 100

def make_image(g_est, gmin, gmax, g, N,L, start, T, im_number, name):
    fig = plt.figure(dpi=150, figsize=(12,3),frameon=False)
    fig.suptitle("b: "+str(T)[:8]+"\nt: "+str(time.clock()-start)[:5] , x=0.95, y = 0.5)
    g_imshow1 = []
    for l in range(L-1):
        g_imshow1.append( g[l*(N-1) : l*(N-1) + N-1] +[gmin] )
        g_imshow1.append( g[L*(N-1) + l*N : L*(N-1) + l*N + N] )
    g_imshow1.append( g[(L-1)*(N-1) : L*(N-1)] + [gmin])
    g_imshow1.append( [gmax]*N)
    fig.add_subplot(131)
    plt.imshow(g_imshow1, interpolation="nearest", cmap="hot")
    
    g_imshow2 = []
    for l in range(L-1):
        g_imshow2.append( g_est[l*(N-1) : l*(N-1) + N-1] +[gmin] )
        g_imshow2.append( g_est[L*(N-1) + l*N : L*(N-1) + l*N + N] )
    g_imshow2.append( g_est[(L-1)*(N-1) : L*(N-1)] + [gmin])
    g_imshow2.append( [gmax]*N)
    fig.add_subplot(132)
    plt.imshow(g_imshow2, interpolation="nearest", cmap="hot")
    
    g_diff = [ gmin + abs(g_1 - g_2) for g_1,g_2 in zip(g,g_est)]
    g_imshow3 = []
    for l in range(L-1):
        g_imshow3.append( g_diff[l*(N-1) : l*(N-1) + N-1] +[gmin] )
        g_imshow3.append( g_diff[L*(N-1) + l*N : L*(N-1) + l*N + N] )
    g_imshow3.append( g_diff[(L-1)*(N-1) : L*(N-1)] + [gmin])
    g_imshow3.append( [gmax]*N)
    fig.add_subplot(133)
    plt.imshow(g_imshow3, interpolation="nearest",cmap="hot")
    
    plt.savefig(name+str(im_number)+".png", bbox_inches="tight", pad_inches=0.0,
        frameon=False)

def make_image2(g_est, gmin, gmax, g, N,L, start, T, im_number, name):
    fig = plt.figure(dpi=150, figsize=(12,3),frameon=False)
    fig.suptitle("b: "+str(T)[:8]+"\nt: "+str(time.clock()-start)[:5] , x=0.95, y = 0.5)
    g_imshow1 = []
    for l in range(L):
        g_imshow1.append(g[l*N:(l+1)*N])
    g_imshow1.append( [gmin]*N)
    g_imshow1.append( [gmax]*N)
    fig.add_subplot(131)
    plt.imshow(g_imshow1, interpolation="nearest", cmap="hot")
    
    g_imshow2 = []
    for l in range(L):
        g_imshow2.append(g_est[l*N:(l+1)*N])
    g_imshow2.append( [gmin]*N)
    g_imshow2.append( [gmax]*N)
    fig.add_subplot(132)
    plt.imshow(g_imshow2, interpolation="nearest", cmap="hot")
    
    g_diff = [ gmin + abs(g_1 - g_2) for g_1,g_2 in zip(g,g_est)]
    g_imshow3 = []
    for l in range(L):
        g_imshow3.append(g_diff[l*N:(l+1)*N])
    g_imshow3.append( [gmin]*N)
    g_imshow3.append( [gmax]*N)
    fig.add_subplot(133)
    plt.imshow(g_imshow3, interpolation="nearest",cmap="hot")
    
    plt.savefig(name+"2-"+str(im_number)+".png", bbox_inches="tight", pad_inches=0.0,
        frameon=False)



"""############################################################################

 ###
#  
 ###
    #
####

############################################################################"""


def shake(f,x0,xl,xh,b,n):
    M = len(x0)
    
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = copy.copy(E0)
    
    swaps = 0
    
    for i in range(n*M):
        x1[np.random.randint(0,M)] = np.random.uniform(xl,xh)
        E1 = f(x1)
        if np.random.uniform(0.0,1.0) <= np.exp( - b*(E1 - E0) ):
            x0 = copy.copy(x1)
            E0 = copy.copy(E1)
            swaps += 1
        else:
            x1 = copy.copy(x0)
    return E0, x0

def metropolis_standard(f, x, x_min, x_max, n, g, N,L, start):
    E = f(x)
    b = 1.0/E
    #while E < epsilon:
    count = 0
    im_count = 1
    while b < 5000:
        E, x = shake(f,x,x_min,x_max, b, n)
        b = b*1.1
        count += 1
        if count%20 == 0:
            make_image(x,x_min,x_max, g, N,L, start, b, im_count, "standard")
            im_count += 1
    return x

"""############################################################################

####
#   #
#   #
#   #
####

############################################################################"""

def shake_demanding(f,x0,xl,xh,b,n):
    M = len(x0)
    
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = copy.copy(E0)
    
    swaps = 0
    
    while swaps < n*M:
        x1[np.random.randint(0,M)] = np.random.uniform(xl,xh)
        E1 = f(x1)
        if np.random.uniform(0.0,1.0) <= np.exp( - b*(E1 - E0) ):
            x0 = copy.copy(x1)
            E0 = copy.copy(E1)
            swaps += 1
        else:
            x1 = copy.copy(x0)
    return E0, x0

def metropolis_demanding(f, x, x_min, x_max, n, g, N,L, start):
    E = f(x)
    b = 1.0/E
    #while E < epsilon:
    count = 0
    im_count = 1
    while b < 5000:
        E, x = shake_demanding(f,x,x_min,x_max, b, n)
        b = b*1.1
        count += 1
        if count%20 == 0:
            make_image(x,x_min,x_max, g, N,L, start, b, im_count, "demanding")
            plt.show()
            print b
            im_count += 1
    return x

"""############################################################################

#
#
#
#
#####

############################################################################"""

    
def shake_linewise(f,x0,xl,xh,b,n, seps):
    M = len(x0)
    
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = copy.copy(E0)
    
    swaps = 0
    
    while swaps < n*M:
        groups = range(len(seps)-1)
        np.random.shuffle(groups)
        for i in groups:
            sep_size = seps[i+1] - seps[i]
            sep_swaps = 0
            while sep_swaps < sep_size:
                members = range(seps[i],seps[i+1])
                np.random.shuffle(members)
                for mem in members:
                    x1[mem] = np.random.uniform(xl,xh)
                    E1 = f(x1)
                    if np.random.uniform(0.0,1.0) <= np.exp( - b*(E1 - E0) ):
                        x0[mem] = x1[mem]
                        E0 = copy.copy(E1)
                        swaps += 1
                        sep_swaps += 1
                    else:
                        x1[mem] = x0[mem]
    return E0, x0

def metropolis_linewise(f, x, x_min, x_max, n, g, N,L, start, seps):
    E = f(x)
    b = 1.0/E
    #while E < epsilon:
    count = 0
    im_count = 1
    while b < 5000:
        E, x = shake_linewise(f,x,x_min,x_max, b, n, seps)
        b = b*1.1
        count += 1
        if count%20 == 0:
            make_image(x,x_min,x_max, g, N,L, start, b, im_count, "linewise")
            plt.show()
            print b
            im_count += 1
    return x

"""############################################################################

 ###    ###
#      #   
 ###   #
    #  #   
####    ###

############################################################################"""
    
def shake_schedule(f,x0,xl,xh,b,n, seps, E_min, x_min, x_vals, x_sigs):
    M = len(x0)
    
    x1 = copy.copy(x0)
    
    E0 = f(x0)
    E1 = copy.copy(E0)
    
    swaps = 0
    E_list = [E0]
    while swaps < n*M:
        groups = range(len(seps)-1)
        np.random.shuffle(groups)
        for i in groups:
            sep_size = seps[i+1] - seps[i]
            sep_swaps = 0
            while sep_swaps < sep_size:
                members = range(seps[i],seps[i+1])
                np.random.shuffle(members)
                for mem in members:
                    x1[mem] = np.random.uniform( max([xl,x1[mem]-8*x_sigs[mem]]),
                                                 min([xh,x1[mem]+8*x_sigs[mem]]))
                    E1 = f(x1)
                    if np.random.uniform(0.0,1.0) <= np.exp( (b*E0)**2 - (b*E1)**2 ):
                        x0[mem] = x1[mem]
                        E0 = copy.copy(E1)
                        swaps += 1
                        sep_swaps += 1
                        x_vals[mem].pop(0)
                        x_vals[mem].append(x1[mem])
                        if E1 < E_min:
                            E_min = copy.copy(E0)
                            x_min[mem] = x1[mem]
                    else:
                        x1[mem] = x0[mem]
        E_list.append(E0)
    return E0, x0, E_min, x_min, np.std(E_list), np.average(E_list)

def metropolis_schedule(f, x, xl, xh, n, g, N,L, start, seps):
    E = f(x)
    b = 1.0/E
    E_min = copy.copy(E)
    x_min = copy.copy(x)
    b_res = copy.copy(b)
    x_vals = []
    x_sigs = []
    for i in range( 2*N*L - N - L ):
        x_vals.append( list( np.random.uniform( xl, xh, 20 ) )  )
        x_sigs.append( np.std( x_vals[-1] ) )
    #while E < epsilon:
    count = 0
    im_count = 1
    while b < 10000:
        E, x, E_min, x_min, E_sigma, E_mean = shake_schedule(f,x,xl,xh, b, n, seps, E_min, x_min, x_vals, x_sigs)
        b = b/(1 - 0.2*E_sigma/E_mean)
        count += 1
        if b > b_res and False:
            b_res = 10*b
            x = copy.copy(x_min)
            E = copy.copy(E_min)
        if count%40 == 0:
            make_image(x,xl,xh, g, N,L, start, b, im_count, "schedule")
            plt.show()
            print b
            im_count += 1
            print x_sigs[::N]
    make_image(x,xl,xh, g, N,L, start, b, im_count, "schedule")
    return x_min

def metropolis_schedule2(f, x, xl, xh, n, g, N,L, start, seps):
    E = f(x)
    b = 1.0/E
    E_min = copy.copy(E)
    x_min = copy.copy(x)
    b_res = copy.copy(b)
    x_vals = []
    x_sigs = []
    for i in range(N*L ):
        x_vals.append( list( np.random.uniform( xl, xh, 100 ) )  )
        x_sigs.append( np.std( x_vals[-1] ) )
    #while E < epsilon:
    count = 0
    im_count = 1
    while b < 10000:
        E, x, E_min, x_min, E_sigma, E_mean = shake_schedule(f,x,xl,xh, b, n, seps, E_min, x_min, x_vals, x_sigs)
        b = b*np.sqrt(1 + 0.2*E_sigma/E_mean)
        for mem in range(len(x_sigs)):
            x_sigs[mem] = np.std( x_vals[mem] )
        for i in range(len(seps)-1):
            x_sig_max = max(x_sigs[seps[i]:seps[i+1]])
            for j in range(seps[i],seps[i+1]):
                x_sigs[j] = x_sig_max
        count += 1
        if b > b_res and False:
            b_res = 10*b
            x = copy.copy(x_min)
            E = copy.copy(E_min)
        if count%40 == 0:
            make_image2(x,xl,xh, g, N,L, start, b, im_count, "schedule")
            plt.show()
            print b
            im_count += 1
            print x_sigs[::N]
    make_image2(x,xl,xh, g, N,L, start, b, im_count, "schedule")
    return x_min

"""############################################################################

#####
  #
  #
  #
  #

############################################################################"""

def transmat(G_h, G_v, N, L):
    A = np.zeros((N,N))
    U = np.zeros((N,N))
    
    A += - np.diagflat( G_h[L-1], 1 )
    A += - np.diagflat( G_h[L-1], -1 )
    A += np.diagflat( G_h[L-1] + [0.0] )
    A += np.diagflat( [0.0] + G_h[L-1] )
    
    for l in range(1,L):
        U = np.diagflat(G_v[L-1-l])
        A = A.dot(linalg.solve( U + A, U, sym_pos=True , overwrite_b = True))
        
        A += - np.diagflat( G_h[L-1-l], 1 )
        A += - np.diagflat( G_h[L-1-l], -1 )
        A += np.diagflat( G_h[L-1-l] + [0.0] )
        A += np.diagflat( [0.0] + G_h[L-1-l] )
    return A

    
    
"""############################################################################

##
######
##########
##############
##################

############################################################################"""
    
def construct_wave(k, high, low):
    mid = 0.5*high + 0.5*low
    amp = 0.5*high - 0.5*low
    g = []
    for l in range(L)[::-1]:
        g = g + [ mid + amp*np.sin( (n+0.5)*k[0] + l*k[1] ) for n in range(N-1) ] 
    for l in range(L-1)[::-1]:
        g = g + [ mid + amp*np.sin( n*k[0] + (l+0.5)*k[1] ) for n in range(N)   ]
    return g 

def construct_wave2(k, high, low):
    mid = 0.5*high + 0.5*low
    amp = 0.5*high - 0.5*low
    g = []
    for l in range(L)[::-1]:
        g = g + [ mid + amp*np.sin( n*k[0] + l*k[1] ) for n in range(N) ]
    return g 

def g_split(G, N, L):
    G_H = []
    G_V = []
    for l in range(L):
        G_H.append( G[l*(N-1) : l*(N-1) + N-1] )
    for l in range(L-1):
        G_V.append( G[L*(N-1) + l*N : L*(N-1) + l*N + N] )
    return G_H, G_V

def norm1(A, G):
    N = len(A)
    L = (len(G) + N)/(2*N-1)
    GH, GV = g_split(G, N, L)
    return np.linalg.norm(A - transmat(GH, GV, N, L) )**0.5
    
def g_split2(G):
    G_H = []
    G_V = []
    for l in range(L-1):  # R1 + R2  =  g1*g2 / ( g1 + g2 )
        G_H.append( [ G[N*l+n]*G[N*l+n+1]/(G[N*l+n]+G[N*l+n+1]) for n in range(N-1)] )
        G_V.append( [ G[N*l+n]*G[N*(l+1)+n]/(G[N*l+n]+G[N*(l+1)+n]) for n in range(N)] )
    G_H.append( [ G[N*L-N+n]*G[N*L-N+n+1]/(G[N*L-N+n]+G[N*L-N+n+1]) for n in range(N-1)] )
    return G_H, G_V

def norm2(A, G):
    GH, GV = g_split2(G)
    return np.linalg.norm( A - schur(GH, GV, N, L) )
    
    
    
def main():
    np.random.seed(3)
    start = time.clock()
    
    gmin = 1.0
    gmax = 2.0
    
    g = [ 0.5*(gmin+gmax) ]*( 2*N*L - N - L )
    g = construct_wave([0.02,0.2],gmax, gmin)
    g = construct_wave([1,1],gmax, gmin)
    gh, gv = g_split(g, N, L)
    A = schur(gh, gv, N, L)
    
    f = lambda x: norm(A, x)
    
    g_in = list(np.random.uniform(gmin, gmax, 2*N*L - N - L))
    
    stand = False
    deman = False
    linew = False
    sched = True
    
    if stand:
        filename = "standard"
        make_image(g_in, gmin, gmax, g, N,L,start,"0",0, filename)
        plt.show()
        g_est = metropolis_standard(f,g_in,gmin,gmax,100, g, N,L, start)
        make_image(g_est, gmin, gmax, g, N,L,start,"INF","LAST", filename)
        plt.show()
    elif deman:
        filename = "demanding"
        make_image(g_in, gmin, gmax, g, N,L,start,"0",0, filename)
        plt.show()
        g_est = metropolis_demanding(f,g_in,gmin,gmax,20, g, N,L, start)
        make_image(g_est, gmin, gmax, g, N,L,start,"INF","LAST", filename)
        plt.show()
    elif linew:
        filename = "linewise"
        make_image(g_in, gmin, gmax, g, N,L,start,"0",0, filename)
        plt.show()
        seps = [0]
        for l in range(1,L+1):
            seps.append(l*(N-1))
        for l in range(1,L):
            seps.append(L*(N-1)+l*N)
        g_est = metropolis_linewise(f,g_in,gmin,gmax,5, g, N,L, start,seps)
        make_image(g_est, gmin, gmax, g, N,L,start,"INF","LAST", filename)
        plt.show()
    elif sched:
        filename = "schedule"
        make_image(g_in, gmin, gmax, g, N,L,start,"0",0, filename)
        plt.show()
        seps = [0]
        for l in range(1,L+1):
            seps.append(l*(N-1))
        for l in range(1,L):
            seps.append(L*(N-1)+l*N)
        g_est = metropolis_schedule(f,g_in,gmin,gmax,10, g, N,L, start,seps)
        make_image(g_est, gmin, gmax, g, N,L,start,"INF","LAST", filename)
        plt.show()
    error = sum([ (o-e)**2 for o,e in zip(g,g_est)])
    print error
    print time.clock() - start
    
    
    
def main2():
    np.random.seed(3)
    start = time.clock()
    
    gmin = 1.0
    gmax = 2.0
    
    g = [ 0.5*(gmin+gmax) ]*(N*L)
    g = construct_wave([1.0,1.0],gmax, gmin)
    g = construct_wave2([1,1],gmax, gmin)
#    g = [ gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,
#          gmin,gmin,gmin,gmax,gmax,gmax,gmin,gmin,gmin,gmax,gmax,gmax,gmin,gmin,gmin,gmax,gmax,gmax,gmin,gmin,gmin,
#          gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,
#          gmin,gmin,gmin,gmax,gmax,gmin,gmin,gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,
#          gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,
#          gmin,gmin,gmin,gmax,gmax,gmax,gmin,gmin,gmin,gmax,gmax,gmax,gmin,gmin,gmin,gmin,gmax,gmin,gmin,gmin,gmin,
#          gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,gmin,
#         ]
    gh, gv = g_split2(g)
    A = schur(gh, gv, N, L)
    
    f = lambda x: norm2(A, x)
    
    g_in = list(np.random.uniform(gmin, gmax, N*L))
    
    sched = True
    
    if sched:
        filename = "schedule"
        make_image2(g_in, gmin, gmax, g, N,L,start,"0",0, filename)
        plt.show()
        seps = []
        for l in range(L+1):
            seps.append(l*N)
        g_est = metropolis_schedule2(f,g_in,gmin,gmax,10, g, N,L, start,seps)
        make_image2(g_est, gmin, gmax, g, N,L,start,"INF","LAST", filename)
        plt.show()
    error = sum([ (o-e)**2 for o,e in zip(g,g_est)])
    print error
    error = max([ abs(o-e) for o,e in zip(g,g_est)])
    print error
    print time.clock() - start

def test():
    g = [1.0]*(2*N*L - N - L)
    gh, gv = g_split(g, N, L)
    start = time.clock()
    for i in range(1):
        A0 = transmat(gh,gv,N,L)
    print time.clock() - start
    print
    start = time.clock()
    #A = trans(gh,gv,N,L)
    #print time.clock() - start
    print
    start = time.clock()
    for i in range(1):
        A3 = schur(gh,gv,N,L)
    print time.clock() - start
    print
    print A0[0][0] - A0[N-1][N-1]
    print A3[0][0] - A3[N-1][N-1]



if __name__ == "__main__":
    #main2()
    test()
















