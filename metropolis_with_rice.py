# -*- coding: utf-8 -*-

import copy
import time

import pp
import numpy as np
from scipy import linalg
from scipy.stats import norm
import matplotlib.pyplot as plt

def parameters():
    N = 12
    L = 4
    gmin = 1.0
    gmax = 2.0
    pattern = "uniform"
    k = [1.0,1.0]
    
    energy_exp = 2.0 # 2, 1, 0.5
    DtN_calc = "transmat" # "trans", "schur"
    #rectangular = True
    #cross_resistors = False
    #stdd_schedule = True
    narrowing_selection = True
    #linewise_narrowing = True
    
    
    nodal = True
    demand = True
    linewise_sweep = True
    
    b_start = 1.0
    b_stop = 10000.0
    try_stop = 100
    swp_stop = 5
    
    histogram_overlap = 0.9
    scale = abs(norm.ppf(histogram_overlap/2))
    
    return N, L, gmin, gmax, nodal, pattern, k, b_start, b_stop, try_stop, swp_stop, scale, demand, linewise_sweep, energy_exp, DtN_calc, narrowing_selection
    
    
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
    
    #plt.savefig(name+str(im_number)+".png", bbox_inches="tight", pad_inches=0.0,
    #    frameon=False)

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
    g_imshow3.append( [gmin]*N )
    g_imshow3.append( [gmax]*N )
    fig.add_subplot(133)
    plt.imshow(g_imshow3, interpolation="nearest",cmap="hot")
    
    #plt.savefig(name+"2-"+str(im_number)+".png", bbox_inches="tight", pad_inches=0.0,
    #    frameon=False)

def g_construct(high, low, nodal, pattern, k):
    mid = 0.5*high + 0.5*low
    amp = 0.5*high - 0.5*low
    g = []
    if nodal:
        if pattern=="wave":
            sigma = lambda n,l: mid + amp*np.sin( n*k[0] + l*k[1] )
        elif pattern=="random":
            sigma = lambda n,l: np.random.uniform(high,low)
        elif pattern=="uniform":
            sigma = lambda n,l: mid
        elif pattern=="half":
            sigma = lambda n,l: low + max([0, N/2 - 0.5 - n])*2*amp/(N/2 - 0.5 - n)
        elif pattern=="box":
            sigma = lambda n,l: mid + amp - 2*amp*(
                                max([0,(- N/4 + 0.5 + n)*(3*N/4 - 0.5 - n)])*
                                max([0,(- L/4 + 0.5 + l)*(3*L/4 - 0.5 - l)])
                                )/(
                                (- N/4 + 0.5 + n)*(3*N/4 - 0.5 - n)*
                                (- L/4 + 0.5 + l)*(3*L/4 - 0.5 - l)
                                )
            
        for l in range(L)[::-1]:
            g = g + [ sigma(n,l) for n in range(N) ]
            print g
    else:
        if pattern=="wave":
            sigma = lambda n,l,h: mid + amp*np.sin( (n+h)*k[0] + (l+0.5-h)*k[1] )
        elif pattern=="random":
            sigma = lambda n,l,h: np.random.uniform(high,low)
        elif pattern=="uniform":
            sigma = lambda n,l,h: mid
            
        for l in range(L)[::-1]:
            g = g + [ sigma(n,l,0.5) for n in range(N-1) ] 
        for l in range(L-1)[::-1]:
            g = g + [ sigma(n,l,0.0) for n in range(N)   ]
    
    return g 


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
    
    
def trans(G_h, G_v, N, L):
    A_tra = np.zeros((N,N))
    g = 0.0
    """ Horizontal resistors, 0th layer """
    for i in range(N-1):
        g = G_h[L-1][i]
        A_tra[i][i+1] += -g
        A_tra[i+1][i] += -g
        A_tra[i][i] += g
        A_tra[i+1][i+1] += g
    
    for l in range(1, L):
        """ Vertical resistors, lth layer """
        for a in range(N):
            g = G_v[L-1-l][a]
            A = copy.copy(A_tra)
            for i in range(N):
                for j in range(N):
                    A_tra[i][j] = A[i][j] - A[i][a]*A[a][j]/(g + A[a][a])
        """ Horizontal resistors, lth layer """
        for a in range(N-1):
            g = G_h[L-1-l][a]
            A_tra[a][a+1] += -g
            A_tra[a+1][a] += -g
            A_tra[a][a] += g
            A_tra[a+1][a+1] += g
    return A_tra
    
def schur(G_h, G_v, N, L):
    A = np.zeros((N,N))
    B = np.zeros((N,N))
    
    for i in range(N):
        A[i][i] += G_v[L-2][i]
        A[i][i] += ( G_h[L-1] + [0.0] )[i]
        A[i][i] += ( [0.0] + G_h[L-1] )[i]
    for i in range(N-1):
        A[i][i+1] += - G_h[L-1][i]
        A[i+1][i] += - G_h[L-1][i]
    
    
    for l in range(1, L-1):
        B = np.diagflat( G_v[L-1-l] )
        A = - linalg.solve(A, B, sym_pos = True)
        A = B.dot(A)
        for i in range(N):
            A[i][i] += G_v[L-1-l][i] + G_v[L-2-l][i]
            A[i][i] += ( G_h[L-1-l] + [0.0] )[i]
            A[i][i] += ( [0.0] + G_h[L-1-l] )[i]
        for i in range(N-1):
            A[i][i+1] += - G_h[L-1-l][i]
            A[i+1][i] += - G_h[L-1-l][i]
    
    B = np.diagflat( G_v[0] )
    A = - linalg.solve(A, B, sym_pos = True, overwrite_b = False)
    A = B.dot(A)
    for i in range(N):
        A[i][i] += G_v[0][i]
        A[i][i] += ( G_h[0] + [0.0] )[i]
        A[i][i] += ( [0.0] + G_h[0] )[i]
    for i in range(N-1):
        A[i][i+1] += - G_h[0][i]
        A[i+1][i] += - G_h[0][i]
    return A
   
def g_split(G, nodal):
    G_H = []
    G_V = []
    if nodal:
        for l in range(L-1):  # R1 + R2  =  g1*g2 / ( g1 + g2 )
            G_H.append( [ G[N*l+n]*G[N*l+n+1]/(G[N*l+n]+G[N*l+n+1]) for n in range(N-1)] )
            G_V.append( [ G[N*l+n]*G[N*(l+1)+n]/(G[N*l+n]+G[N*(l+1)+n]) for n in range(N)] )
        G_H.append( [ G[N*L-N+n]*G[N*L-N+n+1]/(G[N*L-N+n]+G[N*L-N+n+1]) for n in range(N-1)] )
    else:
        for l in range(L):
            G_H.append( G[l*(N-1) : l*(N-1) + N-1] )
        for l in range(L-1):
            G_V.append( G[L*(N-1) + l*N : L*(N-1) + l*N + N] )
    return G_H, G_V

def energy(N,L,A,g,DtN,nodal):
    gh, gv = g_split(g, nodal)
    
    if DtN == "transmat":
        A_try = transmat(gh,gv,N,L)
    elif DtN == "trans":
        A_try = trans(gh,gv,N,L)
    elif DtN == "schur":
        A_try = schur(gh,gv,N,L)
    
    return np.linalg.norm(A - A_try)
    
def seperators(linewise, nodal):
    if linewise:
        if nodal:
            return [N*i for i in range(L+1)]
        else:
            return ([(N-1)*i for i in range(L+1) ] + 
                    [L*(N-1) + N*i for i in range(L) ] )
    else:
        if nodal:
            return [0, N*L]
        else:
            return [0 , 2*N*L - N - L]


def p_metro(E0, E1, b, q):
    return ( np.random.uniform() < np.exp( (b*E0)**q - (b*E1)**q ) )


def metropolis(N,L,A,g,g0,gl,gh,nodal,b_start,b_stop,try_stop,swp_stop,scale,demand,linewise_sweep, energy_exp, DtN_calc, narrowing_selection):
    start = time.clock()
    
    eval_time = 0.0
    
    E = lambda x: energy(N,L,A,x,DtN_calc,nodal)
    seps = seperators(linewise_sweep, nodal)
    
    if nodal:
        M = N*L
    else:
        M = 2*N*L - L - N
    b = b_start
    try_count = 0
    swp_count = 0
    img_count = 0
    
    g1 = copy.copy(g0)
    E0 = E(g0)
    E1 = E(g1)
    
    E_avg_list = []
    sigma_list = []
    
    dif_g_list = []
    sig_g_list = []
    dif_g_inf_list = []
    sig_g_inf_list = []
    
    b_list = []
    b_print = b
    
    fails_list = []
    tries_list = []
    fails = 0
    tries = 0
    
    
    g_vals = []
    g_sigs = []
    for i in range(N*L ):
        g_vals.append( list( np.random.uniform( gmin, gmax, 100 ) )  )
        g_sigs.append( np.std( g_vals[-1] ) )
    
    while b < b_stop:
        try_count = 0
        swp_count = 0
        fails = 0
        tries = 0
        E_list = []
        dglist = []
        dginflist = []
        while swp_count < swp_stop*M:
            for l in range(len(seps)-1):
                swp_count_line = 0
                while swp_count_line<=N:
                    nodes = range(seps[l], seps[l+1])
                    np.random.shuffle(nodes)
                    for n in nodes:
                        if narrowing_selection:
                            g1[n] = np.random.uniform(max([gl,g1[n]-6*g_sigs[n]]),
                                                        min([gh,g1[n]+6*g_sigs[n]]))
                        else:
                            g1[n] = np.random.uniform(gl,gh)
                        eval_time = eval_time - time.clock()
                        E1 = E(g1)
                        eval_time = eval_time + time.clock()
                        success = p_metro(E0,E1,b,energy_exp)
                        if success:
                            g0[n] = g1[n]
                            E0 = copy.copy(E1)
                            swp_count += 1
                            swp_count_line += 1
                            E_list.append(E0**energy_exp)
                            dglist.append(np.linalg.norm(np.array(g) - np.array(g0) ))
                            dginflist.append(max(abs(np.array(g) - np.array(g0))))
                            g_vals[n].append(g0[n])
                            g_vals[n].pop(0)
                            tries += 1
                        else:
                            g1[n] = g0[n]
                            try_count += 1
                            swp_count += 1
                            fails += 1
                            tries += 1
        fails_list.append(fails)
        tries_list.append(tries)
        E_avg = np.average(E_list)
        sigma = np.std(E_list)
        E_avg_list.append(E_avg)
        sigma_list.append(sigma)
        
        dg_avg     = np.average(dglist)
        dginf_avg  = np.average(dginflist)
        sigma_g    = np.std(dglist)
        sigma_ginf = np.std(dginflist)
        dif_g_list.append(dg_avg)
        dif_g_inf_list.append(dginf_avg)
        sig_g_list.append(sigma_g)
        sig_g_inf_list.append(sigma_ginf)
        b_list.append(b**energy_exp)
        b = b*( 1.0 + scale*sigma/(E_avg*energy_exp))
        g_sigs = [ np.std(g_val) for g_val in g_vals ]
        if b_print <= b:
            b_print = b_print*np.cbrt(10.0)
            print "b = ",b
            make_image2(g0, gmin, gmax, g, N, L, start, b, img_count, "hei")
            plt.show()
            img_count += 1
    
    make_image2(g0, gmin, gmax, g, N, L, start, b, img_count, "hei")
    print "Time spent evaluating function: ", eval_time
    print "Time spent overall: ", time.clock() - start
    print "Percentage of time in function to overall: ", str(100*eval_time/(time.clock() - start))
    
    plt.show()
    plt.plot([ 1.0/t for t in b_list ] , [ E_i for E_i in E_avg_list ])
    plt.show()
    plt.plot([ np.log(1.0/t) for t in b_list ] , [np.log(E_i) for E_i in E_avg_list])
    plt.show()
    print np.polyfit( [ np.log(1.0/t) for t in b_list ], [np.log(E_i) for E_i in E_avg_list], 1)
    plt.plot([ np.log(1.0/t) for t in b_list ] , [np.log(sig) for sig in sigma_list])
    plt.show()
    print np.polyfit( [ np.log(1.0/t) for t in b_list ], [np.log(E_i) for E_i in sigma_list], 1)
    plt.plot(E_avg_list, dif_g_list)
    plt.plot(E_avg_list, dif_g_inf_list,)
    plt.show()
    plt.plot([ np.log(E_i) for E_i in E_avg_list], dif_g_list)
    plt.plot([ np.log(E_i) for E_i in E_avg_list], dif_g_inf_list,)
    plt.show()
    plt.plot([ np.log(abs(np.log(t)))for t in b_list ], [np.log(dif) for dif in dif_g_list])
    plt.plot([ np.log(abs(np.log(t))) for t in b_list ], [np.log(dif) for dif in dif_g_inf_list])
    plt.show()
    plt.plot( range(len(fails_list)) , fails_list )
    plt.show()
    ratio = [ float(fail)/trie for fail,trie in zip (fails_list,tries_list) ]
    plt.plot( range(len(ratio)) , ratio )
    plt.show()
    print np.polyfit( [np.log(abs(np.log(t))) for t in b_list][len(b_list)/2:],
                       [np.log(dif) for dif in dif_g_list][len(b_list)/2:], 1)
    print 0
    
    
if __name__ == "__main__":
    N, L, gmin, gmax, nodal, pattern, k,b_start, b_stop, try_stop, swp_stop, scale,demand,linewise_sweep, energy_exp, DtN_calc, narrowing_selection = parameters()
    g = g_construct(gmin, gmax, nodal, pattern, k)
    gh,gv = g_split(g, nodal)
    A = transmat(gh, gv, N, L)
    g0 = [1.5]*(N*L)
    metropolis(N, L, A, g, g0, gmin, gmax, nodal, b_start, b_stop, try_stop, swp_stop, scale,demand,linewise_sweep, energy_exp, DtN_calc, narrowing_selection)
