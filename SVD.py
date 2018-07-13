import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
from random import randint
import scipy.sparse.linalg as lin
from scipy.sparse import csr_matrix
from copy import deepcopy
import time

# Simulations

def simulation(topK, maxNumberOfIterations):
    # 100 movieLens
    R, Sp = readMatrixFromFile('u.data', ['user_id', 'movie_id', 'rating'])

    # 1m movieLens
    #R, Sp = readMovieLens1m('ratings.dat')

    # 10m movieLens (truncated)
    #R, Sp = readMovieLens10m('ratings10M.dat')

    # Jokes DataSet
    #R, Sp = readJokesDataSet('jesterfinal151cols.xls')
    
    # MovieLens 100k another
    #R, Sp = read100k('ratings100k.csv')

    numberOfSimulations = np.min(R.shape) - 2
    step = int((np.min(R.shape) - 2)/numberOfSimulations)
    numbersOfLatentFactors = [x * 100 for x in range(1, (int)(numberOfSimulations / 100) + 1)]
    numbersOfLatentFactors.append(numberOfSimulations)
    rmses, maxErrors, recommendations, runningTime, Rs = simulateOverK(R, Sp, numbersOfLatentFactors, topK, maxNumberOfIterations)
    plotFunction(numbersOfLatentFactors, rmses, 'RMSE', 'k', 'rmse')
    plotFunction(numbersOfLatentFactors, maxErrors, 'Absolute error', 'k', 'absolute error')
    plotFunction(numbersOfLatentFactors, runningTime, 'Running time', 'k', 'time')
    return rmses, maxErrors, recommendations, runningTime, Rs

def simulateOverK(R, Sp, numbersOfLatentFactors, topK, maxNumberOfIterations):
    userIndex = randint(0, R.shape[0] - 1)
    rmses = []
    maxErrors = []
    recommendationRatios = []
    recommendations = []
    runningTimes = []
    Rf = meanCenterMatrix(R, Sp)
    Rs = []
    for k in numbersOfLatentFactors:
        startTime = time.time()
        Q, S, Vt, iterations = svdIterativeI(R, Rf, Sp, k, maxNumberOfIterations, True)
        runningTime = time.time() - startTime
        Rp = getRp(Q, S, Vt)
        Rs.append(Rp)
        rmses.append(rmse(Rp - R, Sp))
        maxErrors.append(maxError(Rp - R, Sp))
        currentRecommendation = getTopKRecommendations(Rp, Sp, userIndex, topK)
        recommendations.append(currentRecommendation)
        runningTimes.append(runningTime)
    return rmses, maxErrors, recommendations, runningTimes, Rs

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

def read100k(pathToFile):
    ratings = pd.read_csv(pathToFile).values
    m = (int)(max(ratings[:, 0]))
    items = set(ratings[:, 1])
    n = len(items)
    items = list(items)
    R = np.zeros([m, n], dtype=float)
    S = []
    for i in range(ratings.shape[0]):
        userId = (int)(ratings[i, 0] - 1)
        itemId = items.index(ratings[i, 1])
        R[userId, itemId] = ratings[i, 2]
        S.append([userId, itemId])
    return R, S

def readMovieLens1m(pathToFile):
    file = open(pathToFile)
    m = 0
    n = 0
    list = []
    for line in file:
        data = line.split('::')
        userId = (int)(data[0]) - 1
        itemId = (int)(data[1]) - 1
        rating = (int)(data[2])
        m = max(m, userId)
        n = max(n, itemId)
        list.append([userId, itemId, rating])
    m = m + 1
    n = n + 1
    result = np.zeros([m, n], dtype=float)
    S = []
    for i in range(len(list)):
        result[list[i][0], list[i][1]] = list[i][2]
        S.append([list[i][0], list[i][1]])
    return result, S

def readMovieLens10m(pathToFile):
    file = open(pathToFile)
    m = 1500
    n = 2000
    result = np.zeros([m, n], dtype=float)
    for line in file:
        data = line.split('::')
        userId = (int)(data[0]) - 1
        itemId = (int)(data[1]) - 1
        rating = (float)(data[2])
        if ((userId < m) and (itemId < n)):
            result[userId, itemId] = rating
    r = result[~np.all(result == 0.0, axis=1)]
    r = r.T
    r = r[~np.all(r == 0.0, axis=1)]
    result = r.T
    S = []
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if (result[i, j] > 0):
                S.append([i, j])
    return result, S

def readJokesDataSet(pathToFile):
    file = pd.ExcelFile(pathToFile)
    sheet = file.parse()
    m = len(sheet[sheet.columns[0]])
    n = len(sheet.columns)
    R = np.zeros([m, n], dtype = float)
    S = []
    for i in range(m):
        for j in range(n):
            rating = sheet[sheet.columns[j]][i] + 11
            if (rating <= 21):
                R[i, j] = rating
                S.append([i, j])
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
def svdIterativeI(R, Rf, Sp, k, maxNumberOfIterations, isReversed = False):
    iterations = 0
    while (iterations < maxNumberOfIterations):
        Q, S, Vt = svd(Rf, Sp, k, True)
        Rp = getRp(Q, S, Vt)
        Q1, S1, Vt1 = changeColumnDirection(Q, S, Vt)
        Rf = constructNewR(Rp, R, Sp)
        iterations += 1
    if isReversed == False:
        Q, S, Vt = changeColumnDirection(Q, S, Vt)
    return Q, S, Vt, iterations

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
            if (R[i, j] == 0):
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

# Get singular values

def getSingularValues(R, Sp):
    Rf = meanCenterMatrix(R, Sp)
    if (Rf.shape[0] < Rf.shape[1]):
        R_extended = np.matmul(Rf, Rf.T)
    else:
        R_extended = np.matmul(Rf.T, Rf)
    eigenvalues = np.linalg.eigvals(R_extended)
    singularValues = [x ** 0.5 for x in sorted(eigenvalues.real, reverse=True)]
    return singularValues[:np.min(R.shape) - 2]

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