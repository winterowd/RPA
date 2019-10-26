# -*- coding: utf-8 -*-

import argparse
import os
import os.path
import sys
import string
import math
import cmath
import time
import csv
import numpy as np
from scipy.optimize import curve_fit
from pygsl import integrate
from shutil import copyfile
import InterpolationWrappers

# large \omega behavior of \Delta \Pi(\omega, \vec{q}) = \Pi_B(\omega, \vec{q}) - \Pi_0(\omega, \vec{q})
# fixed \vec{q}; real and imaginary parts of \Delta \Pi
def GetDeltaBosonPropagatorAsymptote(X,a,b):
    """takes in value of frequency and fit parameters"""
    """returns the value of the asymptote for large frequency"""
    omega = X
    return a/(omega**2+b)

def checkExistingFile(filename):
    """check if filename exists and if it does create a backup copy"""
    if (os.path.exists(filename)):
        print "Warning: output file ",filename," exists, taking copy"
        backup="%s.bak"%filename
        copyfile(filename, backup)

# get momentum vector: p = n_1 \vec{b}_1 +  n_2 \vec{b}_2
def GetMomHexagonal(n,L):
    """n: vector of integers characterizing external momentum"""
    """L: vector of integers characterizing size of lattice"""
    assert(len(n)==2)
    assert(len(L)==2)
    p = np.zeros(2)
    p[0] = (2*n[0]/L[0]-n[1]/L[1])/np.sqrt(3)
    p[1] = n[1]/L[1]
    return 2*np.pi*p

# get norm of a vector
def GetNorm(x):
    result = 0.
    for i in range(len(x)):
        result += np.abs(x[i])**2
    return np.abs(result)

# nearest neighbor interaction V(q)_{(0);1,2} = \alpha f(q)
# note: V(q)_{(0);2,1} = V^*(q)_{(0);1,2}
def GetBareDummyBosonPropagtor(n, alpha, L):
    """n: vector of integers characterizing external momentum"""
    """alpha: nearest neighbor interaction (eV)"""
    """L: vector of integers characterizing size of lattice"""
    assert(len(L)==2)
    assert(len(n)==2)
    return alpha*ComputeStructureFactorHexagonalNN(n, L)

# get dressed boson propagator for nearest neighbor interaction
# V(q)_{1,2} = V(q)_{(0);1,2}/(1-V(q)_{(0);2,1}\Pi_{1,2})
def GetDressedDummyBosonPropagator(n, omega, kappa, alpha, L):
    """n: vector of integers characterizing external momentum"""
    """omega: external frequency OR energy"""
    """kappa: hopping (eV)"""
    """alpha: nearest neighbor interaction (eV)"""
    """L: vector of integers characterizing size of lattice"""
    assert(len(n)==2)
    assert(len(L)==2)
    bareV = GetBareDummyBosonPropagtor(n,alpha,L)
    return bareV/(1-np.conj(bareV)*GetPolarizationOperatorMomFrequency(n,omega,kappa,L))

# structure factor 
def ComputeStructureFactorHexagonalNN(n, L):
    """n: vector of integers characterizing external momentum"""
    """L: vector of integers characterizing size of lattice"""
    assert(len(n)==2)
    assert(len(L)==2)

    structureFactor = 1.0
    structureFactor += np.exp(1j*2*np.pi*n[0]/L[0])
    structureFactor += np.exp(1j*2*np.pi*(n[0]/L[0]-n[1]/L[1]))

    return np.exp(-1j*(2*np.pi/3.)*(2*n[0]/L[0]-n[1]/L[1]))*structureFactor

# read list of momenta from file (two column, space separated columns)
def GetMomentaList(filename):
    tempList = []
    separator=' '
    if (not os.path.isfile(inputfileMomentum)):
        print "Error: File", inputfileMomentum, " does not exist"
        sys.exit()
    with file(inputfileMomentum, 'rb') as file_obj:
        count=0
        for line in csv.reader(file_obj,
                delimiter=separator,    # Your custom delimiter.
                # quotechar='#',          # character indicating comments in individual fields
                skipinitialspace=True): # Strips whitespace after delimiter.
            if line: # Make sure there's at least one entry.
                vals=map(int,line)
                assert(len(vals)==2)
                assert(-L/2<=vals[0]<L/2)
                assert(-L/2<=vals[1]<L/2)
                tempList.append((vals[0],vals[1]))
            else:
                print "Line", count, "is empty!"
            count=count+1
    return np.copy(tempList)

def OutputMomGrid(filename, momList, observable, precision=16):
    assert(momList.shape[0]==observable.shape[0])
    assert(momList.shape[1]==2)

    checkExistingFile(filename)

    mom_format="%."+str(precision)+"e %."+str(precision)+"e"
    data_format=" %."+str(precision)+"e"
    
    with open(filename, "w") as text_file:
        for i in range(momList.shape[0]):
            if len(observable.shape) > 1:
                text_file.write(mom_format % (momList[i,0], momList[i,1]))
                for data in observable[i,:]:
                    text_file.write(data_format % (data))
                text_file.write("\n")
            else:
                my_format=mom_format+data_format+"\n"
                text_file.write(my_format % (momList[i,0], momList[i,1], observable[i]))

def Output1DGrid(filename, gridList, observable, precision=16):
    assert(gridList.shape[0]==observable.shape[0])
    assert(len(gridList.shape)==1)

    checkExistingFile(filename)

    mom_format="%."+str(precision)+"e"
    data_format=" %."+str(precision)+"e"
    
    with open(filename, "w") as text_file:
        for i in range(gridList.shape[0]):
            if len(observable.shape) > 1:
                text_file.write(mom_format % (gridList[i]))
                for data in observable[i,:]:
                    text_file.write(data_format % (data))
                text_file.write("\n")
            else:
                my_format=mom_format+data_format+"\n"
                text_file.write(my_format % (gridList[i], observable[i]))

# get \delta \Pi(omega, \vec{q}(n))_{1,2} = \Pi_{(B);1,2}(omega, \vec{q}(n)) - \Pi_{(0);1,2}(\vec{q}(n))
def GetDifferenceDummyBosonPropagator(n,omega,kappa,alpha,L):
    """n: vector of integers characterizing external momentum"""
    """omega: external frequency OR energy"""
    """kappa: hopping (eV)"""
    """alpha: nearest neighbor interaction (eV)"""
    """L: vector of integers characterizing size of lattice"""
    assert(len(n)==2)
    assert(len(L)==2)
    return GetDressedDummyBosonPropagator(n,omega,kappa,alpha,L)-GetBareDummyBosonPropagtor(nExtern,alpha,latticeDims)
            
# inplements polarization operator element \Pi_{1,2} using bare fermion Green's functions
def GetPolarizationOperatorMomFrequency(n, omega, kappa, L):
    """n: vector of integers characterizing external momentum"""
    """omega: external frequency OR energy"""
    """kappa: hopping (eV)"""
    """L: vector of integers characterizing size of lattice"""
    assert(len(n)==2)
    assert(len(L)==2)

    prefactor = 0.5/(L[0]*L[1])
    
    result = 0.0
    A = np.zeros(2,dtype=complex)
    B = np.zeros(2,dtype=complex)
    for n1 in range(int(L[0])):
        for n2 in range(int(L[1])):
            nLoop = np.copy([n1,n2])
            nPlusLoop = n+nLoop[:]
            nMinusLoop = n-nLoop[:]
            A[0] = kappa*ComputeStructureFactorHexagonalNN(nLoop,latticeDims)
            A[1] = np.conj(A[0])
            B[0] = ComputeStructureFactorHexagonalNN(nPlusLoop,latticeDims)
            B[1] = ComputeStructureFactorHexagonalNN(nMinusLoop,latticeDims)
            for i in range(2):
                if np.abs(A[i]) > 1.0e-10 and np.abs(B[i]) > 1.0e-10:
                    num = omega**2 + np.abs(A[i])**2 + np.abs(B[i])**2
                    if num > 1.0e-15:
                        #result += np.exp(1j*np.angle(A[i]))*np.exp(1j*np.angle(B[i]))*(np.abs(A[i])+np.abs(B[i]))/num
                        result += (A[i]/np.abs(A[i]))*(B[i]/np.abs(B[i]))*(np.abs(A[i])+np.abs(B[i]))/num
                    else:
                        print "Encountered division by zero in GetPolarizationOperatorKFrequency()!"
    return prefactor*result

# integration factor which appears after change of variables: \omega = w \tan(rw)
# designed to cancel asymptotic behavior of \delta \Pi(\omega, \vec{q})
def GetSigmaRPAIntegrandWeight(r,w):
    """r: integration variable"""
    """w: asymptote parameter"""
    return w**2*(np.tan(r*w)**2+1.)

# inplements RPA self-energy at first order \Sigma_{RPA;(2,1)}
# uses dressed dummy boson propagator and bare fermion propagator
def GetSelfEnergyRPADummyMomFrequency(n, omega, kappa, L, asymptoteParams, nbrRTraining=40, nbrRTest=200, workspaceSize=100, maxBisections=100, epsilonQuad=1.0e-07,
                                      epsilonRelQuad=1.0e-04, useInterpolator=True, verbose=False, findEnergy=False):
    """n: vector of integers characterizing external momentum"""
    """omega: external frequency OR energy"""
    """kappa: hopping (eV)"""
    """L: vector of integers characterizing size of lattice"""
    """asymptoteParams: tensor of fit parameter characterizing asymptote of \delta \Pi at large frequency"""
    """nbrRTaining: number of points for training interpolator for integrand"""
    """nbrRTest: number of points to test integrand"""
    """workspaceSize: GSL integration workspace size"""
    """maxBisections: GSL integration number of subdivisions"""
    """epsilonQuad: absolute error for integration"""
    """epsilonRelQuad: relative error for integration"""
    """useInterpolator: use the interpolation routine or explicit evalaution of \delta \Pi"""
    """verbose: output integrand?"""
    """findEnergy: searching for energy? i.e. omega \to i\omega (Wick rotation)"""
    assert(len(n)==2)
    assert(len(L)==2)
    assert(len(asymptoteParams.shape)==2)
    assert(asymptoteParams.shape[0]==L[0])
    assert(asymptoteParams.shape[1]==L[1])
        
    prefactor = -1/(4*np.pi*L[0]*L[1])
    resultIntegration = np.zeros((int(L[0]),int(L[1]),4))
    work = integrate.workspace(workspaceSize) #create workspace for integration
    bareSigma = GetSelfEnergyBareDummyMom(n,kappa,L)
    result = 0.0
    for n1 in range(int(L[0])): # sum over spatial loop momentum \vec{q}
        for n2 in range(int(L[1])):
            # integrate over q_0 for each loop momentum \vec{q}
            nLoop = np.copy([n1,n2])
            print 'GetSelfEnergyRPADummyMomFrequency nLoop:',nLoop
            pPlusLoop = n+nLoop[:]
            pMinusLoop = n-nLoop[:]
            
            factorPlusLoop = ComputeStructureFactorHexagonalNN(pPlusLoop,L)
            factorMinusLoop = ComputeStructureFactorHexagonalNN(pMinusLoop,L)
            
            w = asymptoteParams[n1,n2] # get parameter which defines change of variables

            assert(w>0)
            rMax = 0.5*np.pi/w

            if useInterpolator:
                rIntegrandTrain = np.linspace(-rMax,rMax,num=nbrRTraining,endpoint=True)
                integrandTrain = np.zeros((nbrRTraining,4))
            
            if verbose:
                rIntegrandTest = np.linspace(-rMax,rMax,num=nbrRTest,endpoint=True)
                integrandData = np.zeros((nbrRTest,4))
                        
            if np.abs(factorPlusLoop) > 1.0e-10:

                if findEnergy:
                    numReal = lambda r: GetSigmaRPAIntegrandWeight(r,w)*((np.abs(factorPlusLoop)**2-omega**2+w*np.tan(r*w))*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).real
                                                                         + 2*omega*w*np.tan(r*w)*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).imag)
                    denom = lambda r: ((1j*omega+w*np.tan(r*w))**2+np.abs(factorPlusLoop)**2)*((-1j*omega+w*np.tan(r*w))**2+np.abs(factorPlusLoop)**2)
                    realIntegrand1 = lambda r: numReal(r)/denom(r)
                    numImag = lambda r: GetSigmaRPAIntegrandWeight(r,w)*((np.abs(factorPlusLoop)**2-omega**2+w*np.tan(r*w))*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).imag
                                                                         - 2*omega*w*np.tan(r*w)*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).real)
                else:
                    # \delta \Pi_{1,2}(q_0, \vec{q}) G_{(0);2,1}(p_0+q_0,\vec{p}+\vec{q})
                    realIntegrand1 = lambda r: GetSigmaRPAIntegrandWeight(r,w)*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).real/((omega+w*np.tan(r*w))**2 + (kappa*np.abs(factorPlusLoop))**2)
                    imagIntegrand1 = lambda r: GetSigmaRPAIntegrandWeight(r,w)*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).imag/((omega+w*np.tan(r*w))**2 + (kappa*np.abs(factorPlusLoop))**2)

                if useInterpolator:
                    for i in range(nbrRTraining):
                        integrandTrain[i,0] = realIntegrand1(rIntegrandTrain[i])
                        integrandTrain[i,1] = imagIntegrand1(rIntegrandTrain[i])
                
                    realIntegrand1Interp = InterpolationWrappers.MyUnivariateSpline(x=rIntegrandTrain,y=integrandTrain[:,0],name="realIntegrand1",boundsError=True,verbose=True)
                    imagIntegrand1Interp = InterpolationWrappers.MyUnivariateSpline(x=rIntegrandTrain,y=integrandTrain[:,1],name="imagIntegrand1",boundsError=True,verbose=True)

                    realIntFuc = integrate.gsl_function(MakeIntFunction(realIntegrand1Interp.EvaluateUnivariateInterpolationPoint), None) #set function to be integrated (real)
                    imagIntFuc = integrate.gsl_function(MakeIntFunction(imagIntegrand1Interp.EvaluateUnivariateInterpolationPoint), None) #set function to be integrated (imag)
                else:
                    realIntFuc = integrate.gsl_function(MakeIntFunction(realIntegrand1), None) #set function to be integrated (real)
                    imagIntFuc = integrate.gsl_function(MakeIntFunction(imagIntegrand1), None) #set function to be integrated (imag)

                flag, resultIntegration[n1,n2,0], error = integrate.qag(realIntFuc, -rMax, rMax, epsilonQuad, epsilonRelQuad, maxBisections, integrate.GAUSS61, work)
                flag, resultIntegration[n1,n2,1], error = integrate.qag(imagIntFuc, -rMax, rMax, epsilonQuad, epsilonRelQuad, maxBisections, integrate.GAUSS61, work)

                if verbose:
                    for i in range(nbrRTest):
                        integrandData[i,0] = realIntegrand1Interp.EvaluateUnivariateInterpolationPoint(rIntegrandTest[i])
                        integrandData[i,1] = imagIntegrand1Interp.EvaluateUnivariateInterpolationPoint(rIntegrandTest[i])

                    Output1DGrid("Integrand1_kappa_%g_alpha_%g_P_%d_%d_Q_%d_%d_omega_%g_L_%d_nbrR_%d"%(kappa,alpha,n[0],n[1],nLoop[0],nLoop[1],omega,L[0],nbrRTest),
                                 rIntegrandTest,integrandData)
                
                result += -kappa*np.conj(factorPlusLoop)*(resultIntegration[n1,n2,0]+1j*resultIntegration[n1,n2,1])
                
            if np.abs(factorMinusLoop) > 1.0e-10:

                if findEnergy:
                    numReal = lambda r: GetSigmaRPAIntegrandWeight(r,w)*((np.abs(factorMinusLoop)**2-omega**2+w*np.tan(r*w))*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).real
                                                                         - 2*omega*w*np.tan(r*w)*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).imag)
                    denom = lambda r: ((1j*omega-w*np.tan(r*w))**2+np.abs(factorMinusLoop)**2)*((-1j*omega+w*np.tan(r*w))**2-np.abs(factorMinusLoop)**2)
                    realIntegrand1 = lambda r: numReal(r)/denom(r)
                    numImag = lambda r: GetSigmaRPAIntegrandWeight(r,w)*((np.abs(factorMinusLoop)**2-omega**2+w*np.tan(r*w))*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).imag
                                                                         + 2*omega*w*np.tan(r*w)*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).real)
                else:
                    # \delta \Pi_{2,1}(q_0, \vec{q}) G_{(0);2,1}(p_0-q_0,\vec{p}+\vec{q})
                    realIntegrand2 = lambda r: GetSigmaRPAIntegrandWeight(r,w)*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).real/((omega-w*np.tan(r*w))**2 + (kappa*np.abs(factorMinusLoop))**2)
                    imagIntegrand2 = lambda r: -GetSigmaRPAIntegrandWeight(r,w)*GetDifferenceDummyBosonPropagator(nLoop,w*np.tan(r*w),kappa,alpha,L).imag/((omega-w*np.tan(r*w))**2 + (kappa*np.abs(factorMinusLoop))**2)

                if useInterpolator:
                    for i in range(nbrRTraining):
                        integrandTrain[i,2] = realIntegrand2(rIntegrandTrain[i])
                        integrandTrain[i,3] = imagIntegrand2(rIntegrandTrain[i])
                
                    realIntegrand2Interp = InterpolationWrappers.MyUnivariateSpline(x=rIntegrandTrain,y=integrandTrain[:,2],name="realIntegrand2",boundsError=True,verbose=True)
                    imagIntegrand2Interp = InterpolationWrappers.MyUnivariateSpline(x=rIntegrandTrain,y=integrandTrain[:,3],name="imagIntegrand2",boundsError=True,verbose=True)

                    realIntFuc = integrate.gsl_function(MakeIntFunction(realIntegrand2Interp.EvaluateUnivariateInterpolationPoint), None) #set function to be integrated (real)
                    imagIntFuc = integrate.gsl_function(MakeIntFunction(imagIntegrand2Interp.EvaluateUnivariateInterpolationPoint), None) #set function to be integrated (imag)
                else:
                    realIntFuc = integrate.gsl_function(MakeIntFunction(realIntegrand2), None) #set function to be integrated (real)
                    imagIntFuc = integrate.gsl_function(MakeIntFunction(imagIntegrand2), None) #set function to be integrated (imag)

                flag, resultIntegration[n1,n2,2], error = integrate.qag(realIntFuc, -rMax, rMax, epsilonQuad, epsilonRelQuad, maxBisections, integrate.GAUSS61, work)
                flag, resultIntegration[n1,n2,3], error = integrate.qag(imagIntFuc, -rMax, rMax, epsilonQuad, epsilonRelQuad, maxBisections, integrate.GAUSS61, work)

                if verbose:
                    for i in range(nbrRTest):
                        integrandData[i,0] = realIntegrand2Interp.EvaluateUnivariateInterpolationPoint(rIntegrandTest[i])
                        integrandData[i,1] = imagIntegrand2Interp.EvaluateUnivariateInterpolationPoint(rIntegrandTest[i])

                    Output1DGrid("Integrand2_kappa_%g_alpha_%g_P_%d_%d_Q_%d_%d_omega_%g_L_%d_nbrR_%d"%(kappa,alpha,n[0],n[1],nLoop[0],nLoop[1],omega,L[0],nbrRTest),
                                 rIntegrandTest,integrandData)

                result += -kappa*np.conj(factorMinusLoop)*(resultIntegration[n1,n2,2]+1j*resultIntegration[n1,n2,3])

    if verbose:
        print 'resultIntegration:', resultIntegration
        
    return prefactor*result+bareSigma

def MakeIntFunction(f):
    """takes in a scalar y and a function of one variable f(x)"""
    """returns a lambda function of two variables (x,z) which takes the values f(x)"""
    """this format is needed for GSL integration routines"""
    return lambda x,z: f(x)

# inplements self-energy at first order \Sigma_{0;(2,1)}
# uses bare dummy boson propagator and bare fermion propagator
def GetSelfEnergyBareDummyMom(n, kappa, L):
    """n: vector of integers characterizing external momentum"""
    """kappa: hopping (eV)"""
    """L: vector of integers characterizing size of lattice"""
    assert(len(n)==2)
    assert(len(L)==2)

    result = 0
    for n1 in range(int(L[0])):
        for n2 in range(int(L[1])):
            nLoop = np.copy([n1,n2])
            pPlusLoop = n[:]+nLoop[:]
            pMinusLoop = n[:]-nLoop[:]
            factorPlusLoop = ComputeStructureFactorHexagonalNN(pPlusLoop,latticeDims)
            factorMinusLoop = ComputeStructureFactorHexagonalNN(pMinusLoop,latticeDims)
            V12 = GetBareDummyBosonPropagtor(nLoop,alpha,latticeDims)
            V21 = np.conj(V12)
            if np.abs(factorPlusLoop) > 1.0e-15:
                result += V12*np.conj(factorPlusLoop)/np.abs(factorPlusLoop)
            if np.abs(factorMinusLoop) > 1.0e-15:
                result += V21*np.conj(factorMinusLoop)/np.abs(factorMinusLoop)
            
    return -result/(4.*L[0]*L[1])
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='compute first-order self energy for graphene in RPA approximation')
    parser.add_argument("-L",'--L', type=int, nargs='?', default=100, help='extent of hexagonal lattice')
    parser.add_argument("-k",'--kappa', type=float, nargs='?', default=2.7, help='nearest neighbor hopping parameter (eV)')
    parser.add_argument("-a",'--alpha', type=float, nargs='?', default=1.0, help='scale for dummy boson propagator (eV)')
    parser.add_argument("-im",'--inputfile_mom', nargs='?', default='', help='input file containing values of external momentum')
    parser.add_argument("-om",'--omega_max', type=float, nargs='?', default=20.0, help='maximum omega for output/fit of quantities (units of hopping)')
    parser.add_argument("-ol",'--omega_lambda', type=float, nargs='?', default=10.0, help='cutoff for integration of difference between \delta \Pi and asymptote (units of hopping)')
    parser.add_argument("-nff",'--nbr_freq_fit', type=int, nargs='?', default=100, help='numer of frequencies for fit grids of quantities')
    parser.add_argument("-nfo",'--nbr_freq_output', type=int, nargs='?', default=10, help='numer of frequencies for output of quantities on momentum grid')

    args = parser.parse_args()
    L = args.L
    assert(L%2==0)
    kappa = args.kappa
    alpha = args.alpha
    inputfileMomentum = args.inputfile_mom
    omegaMax = kappa*args.omega_max
    omegaLambda = kappa*args.omega_lambda
    nbrFreqFit = args.nbr_freq_fit
    nbrFreqOutput = args.nbr_freq_output

    externMomenta = GetMomentaList(inputfileMomentum)
    omegaOutput = np.linspace(0,omegaMax,nbrFreqOutput)
    assert(omegaMax>omegaLambda)
    omegaFit = np.linspace(omegaLambda,omegaMax,nbrFreqFit)
    
    # precompute several structures and constants
    nbrMomenta = externMomenta.shape[0]
    prefactorSum = 1./(4.*L**2)
    latticeDims = np.zeros(2)
    latticeDims = latticeDims + L

    # data structures for output purposes
    energy = np.zeros(nbrMomenta)
    barePropagator = np.zeros((nbrMomenta,2))
    dressedPropagator = np.zeros((nbrMomenta,2))
    dressedPropagatorOmega = np.zeros((nbrFreqOutput,2))
    differencePropagatorFitDataOmega = np.zeros((nbrFreqFit,4)) # contains data for fit (_,0-2) (real, imag, and abs) and values of fit function (_,3) (abs only)
    SigmaRPAMomFrequency = np.zeros((nbrMomenta,nbrFreqOutput,2))
    SigmaBareMom = np.zeros((nbrMomenta,2))

    #parameters, covariance matrices, and errors for fits: real and imaginary parts for each (n1,n2)
    fit_params = np.zeros((L,L,2),dtype=float) 
    fit_cov = np.zeros((L,L,2,2),dtype=float) # can't estimate without error
        
    for n1 in range(L):
        for n2 in range(L):
            nExtern = np.copy([n1,n2])
            for j in range(nbrFreqFit):
                differenceFit = GetDifferenceDummyBosonPropagator(nExtern,omegaFit[j],kappa,alpha,latticeDims)
                differencePropagatorFitDataOmega[j,0] = differenceFit.real
                differencePropagatorFitDataOmega[j,1] = differenceFit.imag
                differencePropagatorFitDataOmega[j,2] = np.abs(differenceFit)
            fit_params[n1,n2,:], fit_cov[n1,n2,:,:] = curve_fit(GetDeltaBosonPropagatorAsymptote,omegaFit,differencePropagatorFitDataOmega[:,2],
                                                                method='trf',bounds=((-np.inf,0),(np.inf,np.inf))) # fit absolute value
            assert(fit_params[n1,n2,1]>0.) # make sure w^2 is positive!
            #for j in range(nbrFreqFit):
                #differencePropagatorFitDataOmega[j,3] = GetDeltaBosonPropagatorAsymptote(omegaFit[j],fit_params[n1,n2,0],fit_params[n1,n2,1])
                
            #Output1DGrid("FitDifferenceDummyBosonPropagatorOmega_kappa_%g_alpha_%g_Q_%d_%d_L_%d_nbrOmega_%d"%(kappa,alpha,nExtern[0],nExtern[1],L,nbrFreqOutput),
                         #omegaFit,differencePropagatorFitDataOmega)
    print 'w:',np.sqrt(fit_params[:,:,1])

    for i in range(nbrMomenta):
        sigmaBare = GetSelfEnergyBareDummyMom(externMomenta[i,:],kappa,latticeDims)
        SigmaBareMom[i,0] = sigmaBare.real
        SigmaBareMom[i,1] = sigmaBare.imag
    OutputMomGrid("SigmaBare_alpha_%g_kappa_%g_L_%d_nbrP_%d"%(alpha,kappa,L,nbrMomenta),externMomenta,SigmaBareMom)
    
    for i in range(nbrFreqOutput):
        for j in range(nbrMomenta):
            sigmaRPA = GetSelfEnergyRPADummyMomFrequency(externMomenta[j,:],omegaOutput[i],kappa,latticeDims,np.sqrt(fit_params[:,:,1]))
            SigmaRPAMomFrequency[j,i,0] = sigmaRPA.real
            SigmaRPAMomFrequency[j,i,1] = sigmaRPA.imag
            print 'nExtern:',externMomenta[j,:],'omega:',omegaOutput[i],'SigmaRPA:',sigmaRPA
        OutputMomGrid("SigmaRPA_alpha_%g_kappa_%g_omega_%g_L_%d_nbrP_%d"%(alpha,kappa,omegaOutput[i],L,nbrMomenta),externMomenta,SigmaRPAMomFrequency[:,i,:])
    sys.exit()

    # TODO: search using Newton-Raphson method
    # search for roots of: E^2 - |-\kappa f^*(p) - \Sigam_{(RPA}_{2,1}(iE, \vec{p})|^2 = 0, fixed \vec{p}
        
    # print out result for dispersion relation
    for i in range(nbrMomenta):
        energy[i] = np.abs(-kappa*np.conj(ComputeStructureFactorHexagonalNN(externMomenta[i,:],latticeDims))+GetSelfEnergyBareDummyMom(externMomenta[i,:],kappa,latticeDims))
        propagator = GetBareDummyBosonPropagtor(externMomenta[i,:],alpha,latticeDims)
        barePropagator[i,0] = propagator.real
        barePropagator[i,1] = propagator.imag
    OutputMomGrid("BareFirstOrderDispersion_alpha_%g_kappa_%g_L_%d_nbrP_%d"%(alpha,kappa,L,nbrMomenta),externMomenta,energy)
    OutputMomGrid("BareDummyBosonPropagator_alpha_%g_L_%d_nbrP_%d"%(alpha,L,nbrMomenta),externMomenta,barePropagator)
    
    # output dressed propagator on momentum grid for several values of the frequency
    for i in range(nbrFreqOutput):
        for j in range(nbrMomenta):
            propagator = GetDressedDummyBosonPropagator(externMomenta[j,:],omegaOutput[i],kappa,alpha,latticeDims)
            dressedPropagator[j,0] = propagator.real
            dressedPropagator[j,1] = propagator.imag
        OutputMomGrid("DressedDummyBosonPropagator_alpha_%g_omega_%g_L_%d_nbrP_%d"%(alpha,omegaOutput[i],L,nbrMomenta),externMomenta,dressedPropagator)

    # for fixed \vec{q}, output \Pi_RPA(q_0, \vec{q})
    for i in range(nbrMomenta):
        for j in range(nbrFreqOutput):
            propagator = GetDressedDummyBosonPropagator(externMomenta[i,:],omegaOutput[j],kappa,alpha,latticeDims)
            dressedPropagatorOmega[j,0] = propagator.real
            dressedPropagatorOmega[j,1] = propagator.imag
        Output1DGrid("DressedDummyBosonPropagatorOmega_alpha_%g_P_%d_%d_L_%d_nbrP_%d"%(alpha,externMomenta[i,0],externMomenta[i,1],L,nbrMomenta),
                     omegaOutput,dressedPropagatorOmega)
