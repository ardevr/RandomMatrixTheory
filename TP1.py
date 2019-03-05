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


#%% Séparation exacte du spectre


def plot_eigenvalues(N, n, c, R, finite = True) :
    
    if finite :
        X = npr.randn(N,n)
    else : 
        X = 1/np.sqrt(3)*npr.standard_t(size = (N,n), df = 3)
        
    M = 1/n*np.dot(np.sqrt(R),np.dot(X,np.dot(X.T,np.sqrt(R))))
    eigenvalues, _ = np.linalg.eig(M)
    eigenvalues = np.real(eigenvalues)
    plt.figure(0)
    plt.hist(eigenvalues, color = 'red',
                 bins = N//2) 
    plt.legend() 


#%% Graphe de x(t)

def plot_f(la, c_i, c) : 
    
    y = 10**-10
    
    x = np.linspace(0.4, 1.1, 10**2)
    f = [1 + 1j]
    for xx in x : 
        func = lambda t : 1/(-(xx + 1j*y) + c*np.sum(c_i*la/(1+la*t)))
        t = fixed_point(func,x0 = f[-1])
        f.append(1/np.pi*np.imag(t))
    f = np.array(f[1:])
    integral_f = np.sum(f[:-1]*(x[1:] - x[:-1]))
    plt.figure(0)
    plt.plot(x, f/integral_f, color = 'green', label = 'densité f')
    plt.legend()

def plot_x(la, c_i, c) :
    non_admissible_values = [0] + list(-1/la)
    non_admissible_values.sort()
    plt.figure(2)
    non_admissible_values = [-3] + non_admissible_values + [-10**-3]
    x = lambda t : -1/t + c*np.sum(c_i*la/(1+la*t))
    for i in range(len(non_admissible_values) -1) :
        t = np.linspace(non_admissible_values[i], non_admissible_values[i+1], 10**3)
        x_t = np.array([x(tt) for tt in t])
        plt.plot(t,x_t, color = 'blue')
    plt.axis([-3,-10**-3,-1,1])
    # non_admissible_values = non_admissible_values[1:-1]
    # plt.plot(non_admissible_values, 100*np.ones(len(non_admissible_values)),color = 'black')
    plt.show()


#%% Estimation des lambda_i
    

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

def plot_histogram(R, N, n, c_i, la, low_N = 50, high_N = 750, nb_repetitions = 50, MC = 10) :
    
    estimated_las = np.zeros((nb_repetitions,len(c_i)))
    for j in (range(nb_repetitions)) :
        X = npr.randn(N,n)
        estimated_las[j], _, _ = estimate_la(X, R, N, n, c_i)
    plt.figure(4)
    plt.clf()
    temp = np.ravel(estimated_las)
    plt.hist(temp, color = 'cyan', bins = nb_repetitions//2)
    plt.show()
    for k in range(len(la)) : 
        error = np.mean((la[k] - estimated_las[:,k])**2)
        print('Erreur moyenne sur la valeur propre {} : '.format(k+1),error)
    c = N/n
    Ns = np.linspace(low_N, high_N, nb_repetitions, dtype = 'int')
    ns = Ns/c
    error = np.zeros((nb_repetitions, len(c_i)))
    error_naive = 1*error
    temp_i = [0] +  list(c_i)[:-1] + [1]
    for i in tqdm(range(len(Ns))) : 
        N,n = Ns[i], int(ns[i])
        # print('c = ', N/n)
        R = np.eye(N)
        R[:int(c_i[0]*N),:int(c_i[0]*N)] = 1/2*np.eye(int(c_i[0]*N))
        temp_error = np.zeros((MC,len(la)))
        for k in range(MC) : 
            X = npr.randn(N,n)
            estimated_la, _, _ = estimate_la(X, R, N, n, c_i)
            temp_error[k] = np.array([(la[0] - estimated_la[0])**2, (la[1] - estimated_la[1])**2])
        error[i] = np.mean(temp_error, axis = 0)
        M  = 1/n*np.dot(np.sqrt(R),np.dot(X,np.dot(X.T,np.sqrt(R))))
        emp_la, _ = np.linalg.eig(M)
        emp_la = np.real(emp_la)
        naive_la = []
        for j in range(len(temp_i) - 1) : 
            low = int(temp_i[j]*N)
            high = low + int((temp_i[j+1] - temp_i[j])*N)
            sv = np.linspace(low, high - 1, high - low, dtype = 'int')
            naive_la.append(np.mean(emp_la[sv]))
        error_naive[i] = np.array([(la[0] - naive_la[0])**2, (la[1] - naive_la[1])**2])
    plt.figure(5)
    plt.clf()
    plt.semilogy(Ns, error[:,0], color = 'green', label = 'MSE la_1', marker = '^')
    plt.semilogy(Ns, error[:,1], color = 'red', label = 'MSE la_2', marker = 'o')
    print(error)
    plt.semilogy(Ns, error_naive[:,0], color = 'blue', label = 'MSE Naïve Method la_0', marker = '+')
    plt.semilogy(Ns, error_naive[:,1], color = 'magenta', label = 'MSE Naïve Method la_1', marker = '.')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.show()
    print(error_naive)

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


def plot_fix_N(R, N, la, low_c = 10**-3, high_c = 0.12, nb_repetitions = 200, MC = 10) : 
    
    cs = np.linspace(low_c, high_c, nb_repetitions)
    ns = (N/cs).astype('int')
    error = np.zeros((nb_repetitions, len(c_i)))
    for i in tqdm(range(len(ns))) : 
        n = int(ns[i])
        # print('c = ', N/n)
        temp_error = np.zeros((MC,len(la)))
        for k in range(MC) : 
            X = npr.randn(N,n)
            estimated_la, _, _ = estimate_la(X, R, N, n, c_i)
            temp_error[k] = np.array([(la[0] - estimated_la[0])**2, (la[1] - estimated_la[1])**2])
        error[i] = np.mean(temp_error, axis = 0)
    cs = N/ns  
    plt.figure(50)
    plt.semilogy(cs, error[:,0],color = 'red',label = 'Erreur valeur propre 1',marker = '+')
    plt.semilogy(cs,error[:,1],color = 'blue', label = 'Erreur valeur propre 2', marker = 'o')
    plt.legend()
    

def alternative_estimation() :
    pass
    

    

if __name__  == '__main__' :
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        c = 10**-3
        n = 10**4
        N = int(n*c)
        R = np.eye(N)
        c_i = [0.3, 1 - 0.3]
        la = np.array([1/2,1])
        R[:int(c_i[0]*N),:int(c_i[0]*N)] = 1/2*np.eye(int(c_i[0]*N))
        plot_eigenvalues(N,n,c,R, finite = True)
        plot_f(la, c_i, c)
        plot_x(la,c_i, c)
        X = npr.randn(N,n)
        estimator, _, _ = estimate_la(X, R, N, n, c_i)
        #plot_histogram(R, N, n, c_i, la)
        c_0 = 0.11
        separation_spectre(n)
        plot_fix_N(R, N, la, low_c = 10**-3, high_c = 0.2, nb_repetitions = 200, MC = 10)
        
    