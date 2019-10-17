# -*- coding: utf-8 -*-

import argparse
import os
import os.path
import sys
import string
import math
import cmath
import time
import numpy as np
from shutil import copyfile

def GetLatticeMomentum(n, L, dim=2):
    assert(len(n)==dim)
    assert(len(L)==dim)
    p = np.zeros(dim)
    for i in range(dim):
        assert(L[i]>0)
        p[i] = n[i]/L[i]
    return 2*np.pi*p

def AddLatticeMomentum(n1, n2, L, dim=2):
    assert(len(n1)==dim)
    assert(len(n2)==dim)
    assert(len(L)==dim)
    p = np.zeros(dim)
    for i in range(dim):
        p[i] = (n1[i]+n2[i])/L[i]
    return 2*np.pi*p

def SubtractLatticeMomentum(n1, n2, L, dim=2):
    assert(len(n1)==dim)
    assert(len(n2)==dim)
    assert(len(L)==dim)
    p = np.zeros(dim)
    for i in range(dim):
        p[i] = (n1[i]-n2[i])/L[i]
    return 2*np.pi*p

def GetDummyBosonPropagtor(q, mu=1.0, dim=2):
    assert(mu>0)
    assert(len(q)==dim)
    num = NormSqLatticeMomentum(q,dim) + mu**2
    assert(num!=0.0)
    return num

def NormSqLatticeMomentum(q, dim=2):
    assert(len(q)==dim)
    norm = 0.0
    for i in range(dim):
        norm += q[i]**2
    return norm

def NormLatticeMomentum(q, dim=2):
    assert(len(q)==dim)
    return np.sqrt(NormSqLatticeMomentum(q,dim))

def ComputeStructureFactorHexagonalNN(q):
    assert(len(q)==2)

    # compute explicitly using representation of nearest neighbor vectors
    structureFactor = np.exp(1j*q[0])
    structureFactor += np.exp(-1j*0.5*(q[0]-np.sqrt(3)*q[1]))
    structureFactor += np.exp(-1j*0.5*(q[0]+np.sqrt(3)*q[1]))

    return structureFactor

def ComputePhaseStructureFactorHexagonalNN(q):
    assert(len(q)==2)

    np.

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='compute first-order self energy for graphene in RPA approximation')
    parser.add_argument("-L",'--L', type=int, nargs='?', default=100.0, help='extent of hexagonal lattice')
    parser.add_argument("-m",'--mu', type=float, nargs='?', default=1.0, help='dummy boson propagator mass (debug)')
    parser.add_argument("-k",'--kappa', type=float, nargs='?', default=2.7, help='nearest neighbor hopping parameter')
    parser.add_argument("-im",'--inputfile_mom', nargs='?', default='', help='input file containing values of external momentum')

    args = parser.parse_args()
    L = args.L
    assert(L%2==0)
    mu = args.mu
    kappa = args.kappa
    inputfileMomentum = args.inputfile_mom

    tempList = []
    separator=' '
    if (not os.path.isfile(filename)):
        print "Error: File", filename, " does not exist"
        return
    with file(filename, 'rb') as file_obj:
        line=file_obj.readline()
        for line in csv.reader(file_obj,
                delimiter=separator,    # Your custom delimiter.
                # quotechar='#',          # character indicating comments in individual fields
                skipinitialspace=True): # Strips whitespace after delimiter.
            if line: # Make sure there's at least one entry.
                vals=map(int,line)
                assert(len(vals)==2)
                assert(np.fabs(vals[0])<L/2)
                tempList.append((vals[0],vals[1]))
            else:
                print "Line", count, "is empty!"
            count=count+1
    externMomenta = np.copy(tempList)
    print "mom=",externMomenta

    # precompute several structures and constants
    nbrMomenta = externMomenta.shape[0]
    prefactorSum = 1./(4.*L**2)
    latticeDims = np.zeros(2)
    latticeDims = latticeDims + L

    offDiagonalElements = np.zeros((nbrMomenta,2),dtype=complex)
    
    # Compute (G^{-1}_0 - \Sigma)_{1,2}
    for i in range(nbrMomenta):
        tempMom = GetLatticeMomentum(externMoment[i,:], latticeDims)
        offDiagonalElements[i,0] = -kappa*ComputeStructureFactorHexagonalNN(tempMom)
        for q1 in range(-L/2,L/2):
            for q2 in range(-L/2,L/2):
                nLoop = np.copy([q1,q2])
                loopMom = GetLatticeMomentum(nLoop, latticeDims)
                offDiagonalElements[i,0] += GetDummyBosonPropagtor(loopMom,mu)*np.exp(1j*np.angle(ComputeStructureFactorHexagonalNN(AddLatticeMomentum(externMoment[i,:],nLoop))))
                offDiagonalElements[i,0] += GetDummyBosonPropagtor(loopMom,mu)*np.exp(-1j*np.angle(ComputeStructureFactorHexagonalNN(SubtractLatticeMomentum(externMoment[i,:],nLoop))))


    # Compute (G^{-1}_0 - \Sigma)_{2,1}
    for i in range(nbrMomenta):
        tempMom = GetLatticeMomentum(externMoment[i,:], latticeDims)
        offDiagonalElements[i,1] = -kappa*np.conj(ComputeStructureFactorHexagonalNN(tempMom))
        secondTerm = 0.0
        for q1 in range(-L/2,L/2):
            for q2 in range(-L/2,L/2):
                nLoop = np.copy([q1,q2])
                loopMom = GetLatticeMomentum(nLoop, latticeDims)
                offDiagonalElements[i,1] += GetDummyBosonPropagtor(loopMom,mu)*np.exp(-1j*np.angle(ComputeStructureFactorHexagonalNN(AddLatticeMomentum(externMoment[i,:],nLoop))))
                offDiagonalElements[i,1] += GetDummyBosonPropagtor(loopMom,mu)*np.exp(1j*np.angle(ComputeStructureFactorHexagonalNN(SubtractLatticeMomentum(externMoment[i,:],nLoop))))
