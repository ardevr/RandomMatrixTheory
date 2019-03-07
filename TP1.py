#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:42:41 2019

@author: evrardgarcelon
"""

import numpy as np
import numpy.random as npr
import pylab as plt
from scipy.optimize import fixed_point
import warnings
from tqdm import tqdm
import seaborn as sns


# Séparation exacte du spectre


def plot_eigenvalues(N, n, c, R, finite = True) :
    
    if finite :
        X = npr.randn(N,n)
        #X = npr.exponential(size = (N,n), scale = 1) - 1
    else : 
        X = np.sqrt(0.5/(2.5))*npr.standard_t(size = (N,n), df = 2.1)
        
    M = 1/n*np.dot(np.sqrt(R),np.dot(X,np.dot(X.T,np.sqrt(R))))
    eigenvalues, _ = np.linalg.eig(M)
    eigenvalues = np.real(eigenvalues)
    plt.figure(0)
    plt.hist(eigenvalues, color = 'red',
                 bins = N) 
    plt.legend() 


#  Graphe de x(t)

def plot_f(la, c_i, c) : 
    
    y = 10**-10
    x = np.linspace(0.4, 1.1, 100)
    f = [1 + 1j]
    for xx in x : 
        func = lambda t : 1/(-(xx + 1j*y) + c*np.sum(c_i*la/(1+la*t)))
        t = fixed_point(func,x0 = f[-1])
        f.append(1/np.pi*np.imag(t))
    f = np.array(f[1:])
    integral_f = np.sum(f[:-1]*(x[1:] - x[:-1]))*1.5
    plt.figure(0)
    plt.plot(x, f/integral_f, color = 'green')
    plt.legend()

def plot_x(la, c_i, c) :
    non_admissible_values = [0] + list(-1/la)
    non_admissible_values.sort()
    plt.figure(2)
    non_admissible_values = [-3] + non_admissible_values + [-10**-1]
    x = lambda t : -1/t + c*np.sum(c_i*la/(1+la*t))
    for i in range(len(non_admissible_values) -1) :
        t = np.linspace(non_admissible_values[i], non_admissible_values[i+1], 10**3)
        x_t = np.array([x(tt) for tt in t])
        plt.plot(t,x_t, color = 'blue')
    plt.axis([-3,-10**-3,0,1])
    ylim = (0,1)
    for k in range(len(la)) : 
        plt.vlines(-1/la[k], ylim[0], ylim[1], color = 'red', linestyles = 'dashed')
    plt.show()


# Estimation des lambda_i
    

def estimate_la(X, R, N, n, c_i) : 
    
    M = 1/n*np.dot(np.sqrt(R),np.dot(X,np.dot(X.T,np.sqrt(R))))
    eigenvalues, _ = np.linalg.eig(M)
    eigenvalues = np.real(eigenvalues)
    eigenvalues.sort()
    
    a = np.sqrt(eigenvalues/n)
    eta, _ = np.linalg.eig(np.diag(eigenvalues) - np.outer(a,a))
    eta  = np.real(eta)
    eta.sort()
    
    estimated_la = []
    temp_i = [0] +  list(c_i)[:-1] + [1]
    for i in range(len(temp_i) - 1) :
        
        low = int(temp_i[i]*N)
        high = low + int((temp_i[i+1] - temp_i[i])*N)
        sv = np.linspace(low, high - 1, high - low, dtype = 'int')
        estimated_la.append(np.sum(eigenvalues[sv]-eta[sv])*n/N)
    estimated_la = estimated_la/np.array(c_i)                       
    
    return estimated_la, eigenvalues, eta

def plot_histogram(R, N, n, c_i, la, nb_repetitions = 200) :
    
    estimated_las = np.zeros((nb_repetitions,len(c_i)))
    for j in tqdm(range(nb_repetitions)) :
        X = npr.randn(N,n)
        #X = npr.exponential(size = (N,n), scale = 1) - 1
        estimated_las[j], _, _ = estimate_la(X, R, N, n, c_i)
    plt.figure(4)
    plt.clf()
    temp = N*(estimated_las - la)
    plt.hist(temp[:,0], color = 'green')
    plt.figure(5)
    plt.clf()
    plt.hist(temp[:,1], color = 'red')
    plt.show()
    for k in range(len(la)) : 
        error = np.mean((la[k] - estimated_las[:,k])**2)
        print('Erreur moyenne sur la valeur propre {} : '.format(k+1),error)
        
        
def plot_mse(N, n, c_i, la, low_N = 10, high_N = 100, MC = 10) :
    c = N/n
    Ns = np.linspace(low_N, high_N, dtype = 'int')
    print('Ns = ', Ns)
    ns = Ns/c
    print('ns = ', ns)
    error = np.zeros((len(Ns), len(c_i)))
    error_naive = 1*error
    temp_i = [0] +  list(c_i)[:-1] + [1]
    for i in tqdm(range(len(Ns))) : 
        N,n = Ns[i], int(ns[i])
        print('c = ', N/n)
        R = la[1]*np.eye(N)
        R[:int(c_i[0]*N),:int(c_i[0]*N)] = la[0]*np.eye(int(c_i[0]*N))
        temp_error = np.zeros((MC,len(la)))
        temp_error_naive = 0*temp_error
        for k in range(MC) : 
            X = npr.randn(N,n)
            estimated_la, emp_la, _ = estimate_la(X, R, N, n, c_i)
            temp_error[k] = np.array([(la[0] - estimated_la[0])**2, (la[1] - estimated_la[1])**2])
            naive_la = []
            for j in range(len(temp_i) - 1) : 
                low = int(temp_i[j]*N)
                high = low + int((temp_i[j+1] - temp_i[j])*N)
                sv = np.linspace(low, high - 1, high - low, dtype = 'int')
                naive_la.append(np.mean(emp_la[sv]))
            temp_error_naive[k] = np.array([(la[0] - naive_la[0])**2, (la[1] - naive_la[1])**2])
        error_naive[i] = np.mean(temp_error_naive, axis = 0)
        error[i] = np.mean(temp_error, axis = 0)
        print(estimated_la)
    plt.figure(5)
    plt.clf()
    plt.semilogy(Ns, error[:,0], color = 'green', label = 'MSE la_1', marker = '^')
    plt.semilogy(Ns, error[:,1], color = 'red', label = 'MSE la_2', marker = 'o')
    plt.semilogy(Ns,1/(2*Ns**2),color = 'blue', linestyle = 'dashed')
#    plt.semilogy(Ns, error_naive[:,0], color = 'blue', label = 'MSE Naïve Method la_0', marker = '+')
#    plt.semilogy(Ns, error_naive[:,1], color = 'magenta', label = 'MSE Naïve Method la_1', marker = '.')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.show()

def separation_spectre(n) : 
    plt.figure(20)
    plt.ion()
    cs = np.linspace(10**-2, 0.13, 10)
    Ns = (1*cs)
    Ns = np.array([int(n*c) for c in cs])
    for i in tqdm(range(len(cs))) :
        N,c = Ns[i], cs[i]
        R = np.eye(N)
        R[:int(c_i[0]*N),:int(c_i[0]*N)] = 1/2*np.eye(int(c_i[0]*N))
        X = npr.randn(N,n)
        M = 1/n*np.dot(np.sqrt(R),np.dot(X,np.dot(X.T,np.sqrt(R))))
        eigenvalues, _ = np.linalg.eig(M)
        eigenvalues = np.real(eigenvalues)
        plt.figure(i + 5)
        plt.clf()
        plt.title('c0 = {}'.format(c))
        plt.hist(eigenvalues, color = 'red', bins = N//2)
        plt.show()


def plot_fix_N(R, N, la, low_c = 10**-7, high_c = 1, MC = 10) : 
    
    cs = np.linspace(low_c, high_c, 100)
    print(cs)
    ns = (N/cs)
    error = np.zeros((len(cs), len(c_i)))
    for i in tqdm(range(len(ns))) : 
        n = int(ns[i])
        print('n = ', n)
        temp_error = np.zeros((MC,len(la)))
        for k in range(MC) : 
            X = npr.randn(N,n)
            estimated_la, _, _ = estimate_la(X, R, N, n, c_i)
            temp_error[k] = np.array([(la[0] - estimated_la[0])**2, (la[1] - estimated_la[1])**2])
        error[i] = np.mean(temp_error, axis = 0)
        
    plt.figure(50)
    plt.semilogy(cs, error[:,0],color = 'red',label = 'Erreur valeur propre 1',marker = '+')
    plt.semilogy(cs,error[:,1],color = 'blue', label = 'Erreur valeur propre 2', marker = 'o')
    plt.legend()
    

def alternative_estimation() :
    pass
    

    

if __name__  == '__main__' :
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        c = 10**-2
        n = 1*10**4
        N = int(n*c)
        la = np.array([1/2,1])
        R = la[1]*np.eye(N)
        c_i = [0.3, 1 - 0.3]
        R[:int(c_i[0]*N),:int(c_i[0]*N)] = la[0]*np.eye(int(c_i[0]*N))
#        plot_eigenvalues(N,n,c,R, finite = False)
#        plot_f(la, c_i, c)
#        plot_x(la,c_i, c)
#        X = npr.randn(N,n)
#        estimator, _, _ = estimate_la(X, R, N, n, c_i)
#        plot_histogram(R, N, n, c_i, la)
        c_0 = 0.11
        plot_mse(N, n, c_i, la, low_N = 50, high_N = 500, MC = 20)
#        separation_spectre(n)
 #       plot_fix_N(R, N, la, low_c = 10**-5, high_c = 0.5, MC = 20)
