import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from random import randint
import scipy.sparse.linalg as lin
from scipy.sparse import csr_matrix
from copy import deepcopy

# Simulations

def simulation(topK, maxNumberOfIterations):
    R, Sp = readMatrixFromFile('u.data', ['user_id', 'movie_id', 'rating'])
    numbersOfLatentFactors = range(int(np.min(R.shape) / 10) , np.min(R.shape) - 2)
    rmses, maxErrors, singularValuesRatios, recommendationRatios, recommendations = simulateOverK(R, Sp, numbersOfLatentFactors, topK, maxNumberOfIterations)
    plotFunction(numbersOfLatentFactors, rmses, 'RMSE', 'k', 'rmse')
    plotFunction(numbersOfLatentFactors, maxErrors, 'Max error', 'k', 'max error')
    plotFunction(numbersOfLatentFactors, singularValuesRatios, 'Singular values ratio', 'k', 'singular values ratio')
    plotFunction(numbersOfLatentFactors, recommendationRatios, 'Recommendation ratio', 'k', 'recommendation ratio')
    return rmses, maxErrors, singularValuesRatios, recommendationRatios, recommendations

def simulateOverK(R, Sp, numbersOfLatentFactors, topK, maxNumberOfIterations):
    Q, S, Vt, iterations = svdIterativeI(R, Sp, max(numbersOfLatentFactors), maxNumberOfIterations, True)
    Rp = getRp(Q, S, Vt)
    userIndex = randint(0, R.shape[0] - 1)
    originalRecommendations = set(getTopKRecommendations(Rp, Sp, userIndex, topK))
    rmses = []
    maxErrors = []
    singularValuesRatios = []
    recommendationRatios = []
    recommendations = []
    for k in numbersOfLatentFactors:
        print(k)
        Q, S, Vt, iterations = svdIterativeI(R, Sp, k, maxNumberOfIterations, True)
        Rp = getRp(Q, S, Vt)
        rmses.append(rmse(Rp - R, Sp))
        maxErrors.append(maxError(Rp - R, Sp))
        singularValuesRatios.append(min(np.diag(S)) * 100 / np.max(S))
        currentRecommendation = set(getTopKRecommendations(Rp, Sp, userIndex, topK))
        recommendations.append(currentRecommendation)
        intersectedRecommendations = list(originalRecommendations.intersection(currentRecommendation))
        recommendationRatios.append(len(intersectedRecommendations) * 100 / topK)
    return rmses, maxErrors, singularValuesRatios, recommendationRatios, recommendations

# Constructing prediction matrix

def getRp(Q, S, Vt):
    U = np.matmul(Q, S)
    return np.matmul(U, Vt)

# Reading from file

def readMatrixFromFile(pathToFile, columnNames):
    ratings = pd.read_csv(pathToFile, sep = '\t', names = columnNames, usecols=range(3)).values
    m = max(ratings[:, 0])
    n = max(ratings[:, 1])
    R = np.zeros([m, n])
    S = []
    for i in range(ratings.shape[0]):
        R[ratings[i, 0] - 1, ratings[i, 1] - 1] = ratings[i, 2]
        S.append([ratings[i, 0] - 1, ratings[i, 1] - 1])
    return R, S

# Pure SVD factorization

def svd(R, Sp, k, isReversed = False):
    Q, s, Pt = lin.svds(csr_matrix(R), k, which = 'LM')
    S = np.diag(s)
    if isReversed:
        return Q, S, Pt
    return changeColumnDirection(Q, S, Pt)

# Reversing matrices

def changeColumnDirection(Q, S, Pt):
    newQ = np.zeros(Q.shape)
    newS = np.diag(np.diag(S)[::-1])
    newPt = np.zeros(Pt.shape)
    k = S.shape[0]
    for i in range(k):
        newQ[:, i] = Q[:, k - i - 1]
        newPt[i, :] = Pt[k - i - 1, :]
    return newQ, newS, newPt

# Top-k recommendation

def getTopKRecommendations(Rp, Sp, userIndex, k):
    specifiedRatingsIndeces = []
    for index in Sp:
        if (index[0] == userIndex):
            specifiedRatingsIndeces.append(index[1])
    sorted = list(np.argsort(Rp[userIndex, :])[::-1])
    for index in specifiedRatingsIndeces:
        sorted.remove(index)
    return sorted[:k]

# Iterative approaches for SVD

# Iterative SVD - stoping based on threshold
def svdIterative(R, Sp, k, error, isReversed = False):
    Rf = meanCenterMatrix(R, Sp)
    iterations = 0
    currentError = 2 * error
    while (currentError > error):
        Q, S, Vt = svd(Rf, Sp, k, True)
        Rp = getRp(Q, S, Vt)
        currentError = rmse(R - Rp, Sp)
        Rf = constructNewR(Rp, R, Sp)
        iterations += 1
    if isReversed:
        return Q, S, Vt, iterations
    return changeColumnDirection(Q, S, Vt), iterations

# Iterative SVD - stoping based on number of iterations
def svdIterativeI(R, Sp, k, maxNumberOfIterations, isReversed = False):
    Rf = meanCenterMatrix(R, Sp)
    iterations = 0
    while (iterations < maxNumberOfIterations):
        Q, S, Vt = svd(Rf, Sp, k, True)
        Rp = getRp(Q, S, Vt)
        Q1, S1, Vt1 = changeColumnDirection(Q, S, Vt)
        print(iterations)
        print(Q1.round(4))
        print(S1.round(4))
        print(np.matmul(Q1, S1).round(4))
        print(Vt1.round(4))
        print(getRp(Q1, S1, Vt1).round(4))
        Rf = constructNewR(Rp, R, Sp)
        iterations += 1
        print(Rf.round(4))
    if isReversed:
        return Q, S, Vt, iterations
    return changeColumnDirection(Q, S, Vt), iterations

def constructNewR(newR, oldR, Sp):
    for index in Sp:
        newR[index[0], index[1]] = oldR[index[0], index[1]]
    return newR

# Mean-centering

def getMeanVectorForMatrix(R, Sp):
    means = np.zeros(R.shape[0])
    numbers = np.zeros(R.shape[0])
    for index in Sp:
        means[index[0]] += R[index[0], index[1]]
        numbers[index[0]] += 1
    for i in range(means.shape[0]):
        if numbers[i] > 0:
            means[i] = means[i] / numbers[i]
    return means

def meanCenterMatrix(R, Sp):
    meanVector = getMeanVectorForMatrix(R, Sp)
    Rc = deepcopy(R)
    for i in range(Rc.shape[0]):
        for j in range(Rc.shape[1]):
            if Rc[i, j] == 0:
                Rc[i, j] = meanVector[i]
    return Rc

def getUnspecifiedIndeces(R, Sp):
    result = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if [i, j] in Sp:
                continue
            else:
                result.append([i, j])
    return result

# Error measurements

def rmse(A, Sp):
    error = 0
    for index in Sp:
        error += A[index[0], index[1]] ** 2
    return (error / len(Sp)) ** 0.5

def maxError(A, Sp):
    norm = 0
    for index in Sp:
        norm = max(norm, np.abs(A[index[0], index[1]]))
    return norm

# Computing K

def correlationOfMatrix(R):
    r = R.T
    return np.corrcoef(r)

def getIndexOfMaxElement(A):
    index = np.argmax(np.absolute(A))
    i = int(index / A.shape[0])
    j = index % A.shape[0]
    return [i, j]

def removeItemFromMatrix(A, index):
    A = np.delete(A, index[0], axis=0)
    A = np.delete(A, index[1], axis=1)
    return A

def computeK(R, threshold):
    C = correlationOfMatrix(R)
    D = np.diag(np.diag(C))
    T = C - D
    while (True):
        index = getIndexOfMaxElement(T)
        if abs(T[index[0], index[1]]) > threshold:
            T = removeItemFromMatrix(T, index)
        else:
            return min(min(R.shape) - 2, T.shape[0])

# Plotting

def plotFunction(x, y, title, x_label, y_label):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 7
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(x, y, color='black', marker='o', markersize=2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()