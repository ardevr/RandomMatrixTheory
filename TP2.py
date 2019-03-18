#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:39:19 2019

@author: evrardgarcelon
"""

import numpy as np
import numpy.random as npr
import pylab as plt
from tqdm import tqdm

def gen_A_B(M, q, cl) :
    
    n  = len(q)
    A  = np.zeros((n,n))
    for i in range(n) :
        for j in range(i,n) :
            p = q[i]*q[j]*(1 + M[cl[i],cl[j]]/np.sqrt(n))
            A[i,j] = npr.binomial(1,p)
            A[j,i] = A[i,j]
    
    B =(A - np.outer(q,q))
    return A,B

def alignements(q0, M, c_i, low_n = 1500, high_n = 3000) :
    
    ns = np.linspace(low_n, high_n, 20, dtype = 'int')
    align_1 = []
    align_2 = []
    for i in tqdm(range(len(ns))) :
        n = ns[i]
        q = q0*np.ones(n)
        cl = np.zeros(n)
        cl[n//2: 3*n//4] = 1
        cl[3*n//4 :] = 2
        cl = cl.astype('int')
        c_1 = c_i[0]*n
        c_2 = c_i[1]*n
        j_1 = 1*(cl == 0)
        j_2 = 1*(cl == 1)
        A,B = gen_A_B(M,q,cl)
        eigenval, eigenvec = np.linalg.eig(B/np.sqrt(n))
        eigenval = np.real(eigenval)
        max_eigenval = np.argsort(eigenval)
        max_vec_1 = eigenvec[:,max_eigenval[-1]]
        align_1.append(np.dot(j_1,max_vec_1)**2/c_1)
        align_2.append(np.dot(j_2,max_vec_1)**2/c_2)
    
    return np.array(align_1),np.array(align_2),ns

                

if __name__ == '__main__' :
    
    n = 1500
    K = 3
    c_i = np.array([1/2,1/4,1/4])
    cl = np.zeros(n)
    cl[n//2: ] = 1
    cl = cl.astype('int')
    M = 1*np.eye(K)
    M_to_test = []
    M_to_test.append(M)
    M1 = 1*np.ones((K,K))
    M2 = 1*M1
    for i in range(K) :
        M1[i,i] = 10
        M2[i,i] = 3/2
    M_to_test.append(M1)
    M_to_test.append(M2)

    mode = 'bivalues'
    if mode == 'const' :
        # q = q0
        q0 = 1/2
        q = q0*np.ones(n)
  
    if mode == 'unif' :
        # q uniforme autour de q0
        q0 = 1/2
        large = 1/3
        q = npr.uniform(low = q0 - large, high = q0+large, size = n)
          
    if mode == 'bivalues' :
        qs = np.array([1/3,2/3])
        q = npr.choice(a  = qs, size = n)
        
    colors = ['red', 'blue', 'green', 'magenta', 'cyan']
    for j,M in enumerate([]) :
        i = j + 10*j
        A,B = gen_A_B(M,q,cl)
        eigenval, eigenvec = np.linalg.eig(B/np.sqrt(n))
        eigenval = np.real(eigenval)
        plt.figure()
        plt.hist(eigenval, bins = n//10, color = colors[j])
        plt.show()
        
        ordered_eigenvalues = np.argsort(eigenval)[::-1]
        for k in range(4) :
            plt.figure(1+k)
            plt.clf()
            vec = eigenvec[:,ordered_eigenvalues[k]]
            plt.plot(vec, color = colors[k])
        
   
    M = 10*np.eye(3)
    A,B = gen_A_B(M,q,cl)
    eigenval, eigenvec = np.linalg.eig(B/np.sqrt(n))
    eigenval = np.real(eigenval)
    plt.figure()
    plt.hist(eigenval, bins = n//10, color = colors[0])
    plt.show()
    max_eigenvalue = max(eigenval)
    
    print('Maximum eigenvalue : ', max_eigenvalue)
    temp = c_i*np.diag(M)
    las = temp*q0**2 + (1-q0**2)/(temp)
    print('Expected eigenvalue :', las)
    
    al_1,al_2, ns = alignements(q0, M, c_i)
    
    
    
    
        
    
    
    
    
    
    