# -*- coding: utf-8 -*-

import copy
import time

import pp
import numpy as np
from scipy import linalg
from scipy.stats import norm
import matplotlib.pyplot as plt


#==============================================================================
# Define parameters
#==============================================================================
def parameters():
    N = 12
    L = 4
    gmin = 1.0
    gmax = 2.0
    pattern = "wave"
    k = [1.0,1.0]
    
    energy_exp = 1.0 # 2, 1, 0.5
    DtN_calc = "transmat" # "trans", "schur"
    #rectangular = True
    #cross_resistors = False
    #stdd_schedule = True
    narrowing_selection = True
    narrowing_history = 100
    #linewise_narrowing = True
    
    
    nodal = True
    demand = True
    linewise_sweep = True
    
    b_start = 1.0
    b_stop = 100000.0
    try_stop = 100
    swp_stop = 5
    
    histogram_overlap = 0.95
    scale = abs(norm.ppf(histogram_overlap/2))
    
    #save_file = False
    
    return (N, L, gmin, gmax, nodal, pattern, k, b_start, b_stop, try_stop,
            swp_stop, scale, demand, linewise_sweep, energy_exp, DtN_calc,
            narrowing_selection, narrowing_history)
    
    
#==============================================================================
# Plot resistance heatmap
#==============================================================================
def make_image(g_est, gmin, gmax, g, N,L, start, T, im_number, name, nodal):
    g_diff = [ gmin + abs(g_1 - g_2) for g_1,g_2 in zip(g,g_est)]
    imshow_list = [[],[],[]]
    g_list = [g, g_est, g_diff]
    for l in range(L):
        for i in range(3):
            if nodal:
                imshow_list[i].append(g_list[i][l*N:(l+1)*N])
            elif l != L-1:
                imshow_list[i].append(g_list[i][l*(N-1) : l*(N-1) + N-1] +[gmin] )
                imshow_list[i].append(g_list[i][L*(N-1) + l*N : L*(N-1) + l*N + N] )
    for i in range(3):
        if nodal:
            imshow_list[i].append(   [max(gmax*(i%2),gmin*((i+1)%2))
                                     for i in range(N)]    )
        else:
            imshow_list[i].append( g_list[i][l*(N-1) : l*(N-1) + N-1] +[gmin])
            imshow_list[i].append([max(gmax*(i%2),gmin*((i+1)%2))
                                    for i in range(N)])
    fig = plt.figure(dpi=150, figsize=(12,3),frameon=False)
    fig.suptitle("b: "+str(T)[:str(T).find(".")+2]+
                "\nt: "+str(time.clock()-start)[:5] , x=0.95, y = 0.5)
    
    fig.add_subplot(131)
    plt.imshow(imshow_list[0], interpolation="nearest", cmap="bone")
    fig.add_subplot(132)
    plt.imshow(imshow_list[1], interpolation="nearest", cmap="bone")
    fig.add_subplot(133)
    plt.imshow(imshow_list[2], interpolation="nearest",cmap="hot")
    
    #if save_file:
    #plt.savefig(name+"2-"+str(im_number)+".png", bbox_inches="tight", pad_inches=0.0,
    #    frameon=False)


#==============================================================================
# Construct a list of conductances
#
# Pattern is specified with string, and for waves a vector k=[kx,ky]
# Nodal is a boolean variable, and if it is True a nodal distribution is made.
#
# The position and orientation of conductance  g[n]  is given by an unspecified
# convention.
#==============================================================================
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
            sigma = lambda n,l: low + max([0, N/2 - 0.5-n])*2*amp/(N/2 - 0.5-n)
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
    else:
        if pattern=="wave":
            sigma = lambda n,l,h: mid + amp*np.sin((n+h)*k[0] + (l+0.5-h)*k[1])
        elif pattern=="random":
            sigma = lambda n,l,h: np.random.uniform(high,low)
        elif pattern=="uniform":
            sigma = lambda n,l,h: mid
        elif pattern=="half":
            sigma = lambda n,l: low + max([0, N/2 - 0.5-n])*2*amp/(N/2 - 0.5-n)
            
        for l in range(L)[::-1]:
            g = g + [ sigma(n,l,0.5) for n in range(N-1) ] 
        for l in range(L-1)[::-1]:
            g = g + [ sigma(n,l,0.0) for n in range(N)   ]
    
    return g 


#==============================================================================
# Compute DtN-matrix
#
# Functions take:
#   Lists  G_h, G_v  of Horizontal and  Vertical conductances
#   Integers  N, L  being dimensions of grid
#
# There are three different functions that do the same thing:
#
# transmat| Computes  A  iteratively by the matrix equation
#         |             A'  =  S  +  A (U+A)^-1  U
#         | where   A  is the DtN-map at previous layer,
#         | S  is the matrix accounting for the added v-resistors,
#         | U  is the matrix accounting for the added h-resistors.
#
# trans---| Computes  A  resistor by resistor, starting at the first layer.
#
# schur---| Computes  A  by repeated use of the Schur-complement identity
#         | for DtN-maps on resistor grids.
#==============================================================================
def transmat(G_h, G_v, N, L):
    import numpy as np
    A = np.zeros((N,N))
    U = np.zeros((N,N))
    
    A += - np.diagflat( G_h[L-1], 1 )
    A += - np.diagflat( G_h[L-1], -1 )
    A += np.diagflat( G_h[L-1] + [0.0] )
    A += np.diagflat( [0.0] + G_h[L-1] )
    
    for l in range(1,L):
        U = np.diagflat(G_v[L-1-l])
        A = A.dot(np.linalg.solve( U + A, U))
        
        A += - np.diagflat( G_h[L-1-l], 1 )
        A += - np.diagflat( G_h[L-1-l], -1 )
        A += np.diagflat( G_h[L-1-l] + [0.0] )
        A += np.diagflat( [0.0] + G_h[L-1-l] )
    return A
    
def trans(G_h, G_v, N, L):
    import numpy as np
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
    import numpy as np
    from scipy import linalg
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
    
    
#==============================================================================
# Split a list of conductances into horizontal and vertical
#==============================================================================
def g_series(gi, gj):
    return gi*gj/(gi+gj)
    
def g_split(G, nodal,N,L):
    G_H = []
    G_V = []
    if nodal:
        for l in range(L-1):  # R1 + R2  =  g1*g2 / ( g1 + g2 )
            G_H.append( [ g_series(G[N*l+n],G[N*l+n+1]) for n in range(N-1)] )
            G_V.append( [ g_series(G[N*l+n],G[N*(l+1)+n]) for n in range(N)] )
        G_H.append( [ g_series(G[N*L-N+n],G[N*L-N+n+1]) for n in range(N-1)] )
    else:
        for l in range(L):
            G_H.append( G[l*(N-1) : l*(N-1) + N-1] )
        for l in range(L-1):
            G_V.append( G[L*(N-1) + l*N : L*(N-1) + l*N + N] )
    return G_H, G_V
    

#==============================================================================
# Energy of the system:
# Calculate  A_est  before taking the norm  || A - A_est ||.
#
# (In truth, this is not the actual energy; We have yet to raise it to the
# power  energy_exp)
#==============================================================================
def energy(N,L,A,g,DtN,nodal):
    import numpy as np
    gh, gv = g_split(g, nodal,N,L)
    
    if DtN == "transmat":
        A_try = transmat(gh,gv,N,L)
    elif DtN == "trans":
        A_try = trans(gh,gv,N,L)
    elif DtN == "schur":
        A_try = schur(gh,gv,N,L)
    else:
        return np.zeros((N,N))
    
    return np.linalg.norm(A - A_try)


#==============================================================================
# Make a list of seperator indices
#
# These indices point to the start of different layers in the list  g
#==============================================================================
def seperators(linewise, nodal):
    if linewise:
        if nodal:
            return [N*i for i in range(L+1)]
        else:
            return ([(N-1)*i for i in range(L) ] + 
                    [L*(N-1) + N*i for i in range(L) ] )
    else:
        if nodal:
            return [0, N*L]
        else:
            return [0 , 2*N*L - N - L]


#==============================================================================
# Check if a step is accepted  --  returns boolean
#==============================================================================
def p_metro(E0, E1, b, q, rand):
    return ( rand < np.exp( (b*E0)**q - (b*E1)**q ) )
    

#==============================================================================
# Lehmer random number generator
#==============================================================================
def rng(X):
    X = (X * 48271) % 2147483647
    return float(X)/2147483647, X


#==============================================================================
# METROPOLIS
#==============================================================================
def metropolis(N, L, A, g, g0, gl, gh, nodal, b_start, b_stop, try_stop,
               swp_stop, scale, demand, linewise_sweep, energy_exp, DtN_calc,
               narrowing_selection, narrowing_history):
    start = time.clock()
    
    num = 1
    ber = 48271
    gen = 2**31 - 1
    for i in range(1200):
        num = num*ber%gen
    rand = float(num)/gen
    
    eval_time = 0.0
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
    E0 = energy(N,L,A,g0,DtN_calc,nodal)
    E1 = energy(N,L,A,g1,DtN_calc,nodal)
    
    g_vals = []
    g_sigs = []
    
    for i in range(M):
        g_vals.append( list(np.random.uniform(gmin,gmax,narrowing_history)) )
        g_sigs.append( np.std( g_vals[-1] ) )
    
    #----------------------------+
    # Everything in this box    #|
    # is for data collection    #|
    E_avg_list = []             #|
    sigma_list = []             #|
                                #|
    dif_g_list = []             #|
    sig_g_list = []             #|
    dif_g_inf_list = []         #|
    sig_g_inf_list = []         #|
                                #|
    b_list = []                 #|
    b_print = b                 #|
                                #|
    fails_list = []             #|
    tries_list = []             #|
    fails = 0                   #|
    tries = 0                   #|
    fail_streak_list = []       #|
    b_streak_list = []          #|
    fail_streak = 0             #|
    #----------------------------+
    
    while b < b_stop:
        try_count = 0
        swp_count = 0
        
        # Data ------------------+
        fails = 0               #|
        tries = 0               #|
        fail_streak = 0         #|
        E_list = []             #|
        dglist = []             #|
        dginflist = []          #|
        #------------------------+
        
        while swp_count < swp_stop*M:
            for l in range(len(seps)-1):
                swp_count_line = 0
                while swp_count_line<=seps[l+1]-seps[l]:
                    nodes = range(seps[l], seps[l+1])
                    for n in nodes:
                        rand, num = rng(num)
                        if narrowing_selection:
                            gn = ( (1-rand)*max([gl,g1[n]-8*g_sigs[n]])
                                    + rand*min([gh,g1[n]+8*g_sigs[n]]  ) )
                            g1[n] = gn
                        else:
                            gn = ( (1-rand)*gl + rand*gh )
                            g1[n] = gn
                             
                        time_temp = time.clock()
                        E1 = energy(N,L,A,g1,DtN_calc,nodal)
                        eval_time = eval_time + (time.clock()-time_temp)
                        
                        rand, num = rng(num)
                        success = p_metro(E0,E1,b,energy_exp,rand)
                        
                        if success:
                            g0[n] = g1[n]
                            E0 = copy.copy(E1)
                            swp_count += 1
                            swp_count_line += 1
                            
                            g_vals[n].append(g0[n])
                            g_vals[n].pop(0)
                            
                            # Data collection below
                            E_list.append(E0**energy_exp)
                            dglist.append(np.linalg.norm(np.array(g) - np.array(g0) ))
                            dginflist.append(max(abs(np.array(g) - np.array(g0))))
                            tries += 1
                            fail_streak_list.append(fail_streak)
                            b_streak_list.append(b)
                            fail_streak = 0
                        else:
                            g1[n] = g0[n]
                            try_count += 1
                            swp_count += 1
                            
                            # Data collection below
                            fails += 1
                            tries += 1
                            fail_streak += 1
        
        # Data ------------------+
        fails_list.append(fails)#|
        tries_list.append(tries)#|
        E_avg = np.average(E_list)
        sigma = np.std(E_list)  #|
        E_avg_list.append(E_avg)#|
        sigma_list.append(sigma)#|
                                #|
        dg_avg     = np.average(dglist)
        dginf_avg  = np.average(dginflist)
        sigma_g    = np.std(dglist)
        sigma_ginf = np.std(dginflist)
        dif_g_list.append(dg_avg)
        dif_g_inf_list.append(dginf_avg)
        sig_g_list.append(sigma_g)
        sig_g_inf_list.append(sigma_ginf)
        b_list.append(b**energy_exp)
                                #|
        #------------------------+
        
        g_sigs = [ np.std(g_val) for g_val in g_vals ]
        b = b*( 1.0 + scale*sigma/(E_avg*energy_exp))
        
        if b_print <= b:
            b_print = b_print*np.sqrt(10.0)
            print "b = ",b
            make_image(g0, gmin, gmax, g, N, L, start, b, img_count, "h", nodal)
            plt.show()
            img_count += 1
    
    print "Time spent evaluating energy: ", eval_time
    print "Time spent overall: ", time.clock() - start
    print ("Percentage of time on energy: ",
        str(100*eval_time/(time.clock() - start)))
    
    plt.plot([ 1.0/t for t in b_list ] ,
             [ E_i for E_i in E_avg_list ])
    plt.show()
    plt.plot( [np.log(1.0/t) for t in b_list ] ,
              [np.log(E_i) for E_i in E_avg_list])
    plt.show()
    print np.polyfit(  [np.log(1.0/t) for t in b_list ],
                       [np.log(E_i) for E_i in E_avg_list], 1)
                       
    plt.plot( [np.log(1.0/t) for t in b_list ] ,
              [np.log(sig) for sig in sigma_list])
    plt.show()
    print np.polyfit(  [np.log(1.0/t) for t in b_list ] ,
                       [np.log(E_i) for E_i in sigma_list], 1)
                       
    plt.plot(E_avg_list, dif_g_list)
    plt.plot(E_avg_list, dif_g_inf_list,)
    plt.show()
    plt.plot([ np.log(E_i) for E_i in E_avg_list], dif_g_list)
    plt.plot([ np.log(E_i) for E_i in E_avg_list], dif_g_inf_list,)
    plt.show()
    plt.plot( [np.log(abs(np.log(t)))for t in b_list ],
              [np.log(dif) for dif in dif_g_list] )
    plt.plot( [np.log(abs(np.log(t))) for t in b_list ],
              [np.log(dif) for dif in dif_g_inf_list])
    plt.show()
    plt.plot( range(len(fails_list)) , fails_list )
    plt.show()
    ratio = [ float(fail)/trie for fail,trie in zip (fails_list,tries_list) ]
    plt.plot( range(len(ratio)) , ratio )
    plt.show()
    print np.polyfit( [np.log(abs(np.log(t))) for t in b_list][len(b_list)/2:],
                       [np.log(dif) for dif in dif_g_list][len(b_list)/2:], 1)
    plt.plot( range(len(fail_streak_list)) , fail_streak_list )
    plt.show()
    plt.plot( [ np.log(b_s) for b_s in b_streak_list], fail_streak_list, ".")
    plt.show()
    
    
if __name__ == "__main__":
    # Get parameters
    (N, L, gmin, gmax, nodal, pattern, k, b_start, b_stop, try_stop,
     swp_stop, scale,demand,linewise_sweep, energy_exp, DtN_calc,
     narrowing_selection, narrowing_history) = parameters()
    
    # Prepare the goal distribution, to be approximated
    g = g_construct(gmin, gmax, nodal, pattern, k)
    gh,gv = g_split(g, nodal,N,L)
    A = transmat(gh, gv, N, L)
    
    # Prepare the initial guess
    g0 = []
    if nodal:
        for i in range(N*L):
            g0.append(np.random.uniform(gmin,gmax))
    else:
        for i in range(2*N*L - N - L):
            g0.append(np.random.uniform(gmin,gmax))
    # Call the algorithm
    metropolis(N, L, A, g, g0, gmin, gmax, nodal, b_start, b_stop, try_stop,
               swp_stop, scale,demand,linewise_sweep, energy_exp, DtN_calc,
               narrowing_selection, narrowing_history)