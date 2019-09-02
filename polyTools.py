
# coding: utf-8
'''polyTools module
This module contains tools for calculating various measurements of polymer chains.
'''

# <editor-fold Module Documentation
'''A note on input format
savePoints
----------
savePoints is a list containing information about save points to be used in calculation.
savePoints is always in the (start, end, next) format where
- start (int) is the starting save point.
- end (int) is the last save point (inclusive).
- next (int) is the increment to the next save point.
For example,
savePoints = [0, 10, 2] means the following save points:
0, 2, 4, 6, 8, 10
savePoints = [0, 10, 3] means the following save points:
0, 3, 6, 9

directory
---------
directory is the directory of the data to be used in calculation.
The format is as follows:
/local/whatever/whatever/data/

getEnergies
-----------
Indices of the columns inside energiesv0 file are as follow:
0 ind
1 id
2 E-chi
3 x-chi
4 c-chi
5 E-mu
6 x-mu
7 c-mu
8 E-field
9 x-field
10 c-field
11 E-couple
12 x-couple
13 c-couple
14 E-kap
15 x-kap
16 c-kap
17 E-bend
18 x-bend
19 c-bend
20 E-stretch
21 x-stretch
22 c-stretch
23 E-shear
24 x-shear
25 c-shear
26 E-maierSp
27 x-maierSp
28 c-maierSp
29 E-external
30 x-external
31 c-external
32 E-twoBody
33 x-twoBody
34 c-twoBody
35 E-twist
36 x-twist
37 c-twist
38 E-bind
39 x-bind
40 c-bind
41 E-self
42 x-self
43 c-self
44 E-expl.Bnd
45 x-expl.Bnd
46 c-expl.Bnd
47 E-confine
48 x-confine
49 c-confine
50 E-umbrella
51 x-umbrella
52 c-umbrella
53 E-umbrell2
54 x-umbrell2
55 c-umbrell2
'''
# </editor-fold>

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.integrate as inte
import scipy.signal as sgl
from scipy.spatial import ConvexHull

# ---------- auxiliary functions ----------#

# <editor-fold getNumBeads doc
''' Number of beads in a chain

Returns
-------
N : int
    The number of beads in a chain

Note
----
This code assumes there is only one chain in the simulation.
'''
# </editor-fold>
def getNumBeads(directory: str):
    fname = directory + 'r0v0'
    thisR = np.loadtxt(fname)
    return thisR.shape[0]

# <editor-fold windowAve
''' Moving-point average

Parameters
----------
signal: (N, 2) list
    Signal in time domain
    First column is the time (save point)
    Second column is the signal

Returns
-------
result: (N', 2) np.ndarray
    The window-averaged signal
    First column is the save point
    Second column is the result signal corresponding to the save point

Note
----
For an even-size signal, the resulting windowed signal is delayed by half a time step.
Only valid average values are included in the result. The corresponding time step
is in the same row of the result array.
'''
# </editor-fold>
def windowAve(signal: list, winSize: int):
    resultSize = signal.shape[0] - winSize + 1
    result = np.empty([resultSize, 2])
    window = np.ones(winSize)
    if winSize % 2 == 1:
        # odd window size
        startIndex = int(signal[0, 0] + np.floor(winSize / 2))
        endIndex = int(signal[-1, 0] - np.floor(winSize / 2))
    else:
        # even window size
        startIndex = int(signal[0, 0] + winSize / 2)
        endIndex = int(signal[-1, 0] - winSize / 2 + 1)
    result[:, 0] = signal[startIndex:endIndex+1, 0]
    result[:, 1] = sgl.convolve(signal[:, 1], window, mode='valid') * (1/winSize)
    return result

# <editor-fold autoCorr
''' Autocorrelation

Parameters
----------
signal: (N, 2) list
    Signal in time domain
    First column is the time (save point)
    Second column is the signal

Returns
-------
result:

Note
----
*** Development note ***
Need to check why later half of the auto-convolution signal is the
output autocorrelation.
There might be other definitions of autocorrelation. This function is modified
from
https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
'''
# </editor-fold>
def autoCorr(signal: list):
    result = np.empty(signal.shape)
    result[:, 0] = signal[:, 0]
    autoCorrelation = np.correlate(signal[:, 1], signal[:, 1], mode='full')
    result[:, 1] = autoCorrelation[int(autoCorrelation.shape[0]/2):]
    return result

# <editor-fold autoCorrCB
''' Autocorrelation function, at selected delays
(auto_correlation_vector in the code base)

Parameters
----------
signal: (N) list
    Input signal
delMin: int
    Minimum delay (samples)
delMax: int
    Maximum delay (samples)

Returns
-------
result: (N, 2) np.ndarray
    First column is delta, the delay in terms of number of samples.
    Second column is the corresponding autocorrelation.

Note
----
This is a helper function of autoCorrCBVt
In this function, the definition of autocorrelation is
Rff(delta) = (sum (signal(t) - ave)(signal(t + delta) - ave))/(size * variance)
'''
# </editor-fold>
def autoCorrCB(signal: list, delMin: int, delMax: int):
    if delMin > delMax :
        raise ValueError('delMin > delMax')
    result = np.empty([delMax - delMin + 1, 2])
    for i in range(delMax - delMin + 1):
        thisDel = delMin + i
        thisAuto = autoCorrCB(signal, thisDel)
        result[i, 0] = thisDel
        result[i, 1] = thisAuto
    return result

# <editor-fold autoCorrAtDelCB
''' Autocorrelation at a single delay
(auto_correlation in  the code base)

Parameters
----------
signal: (N) list
    Input signal
delta: int
    Delay (samples)

Returns
-------
result: float
    Autocorrelation at the specified delay

Note
----
This is a helper of autoCorrCB.
In this function, the definition of autocorrelation is
Rff(delta) = (sum (signal(t) - ave)(signal(t + delta) - ave))/(size * variance)
'''
# </editor-fold>
def autoCorrAtDelCB(signal: list, delta: int):
    ave = np.average(signal)
    var = np.var(signal)
    flucPdt = np.empty([signal.shape[0] - delta])
    for i in range(flucPdt.shape[0]):
        flucPdt[i] = (signal[i] - ave) * (signal[i + delta] - ave)
    return np.sum(flucPdt) / ((signal.shape[0] - delta) * var)

# <editor-fold autoCorrCBVt
''' Autocorrelation at selected delays at selected save points
Use the autocorrelation defined in the code base

Parameters
----------
duration: int
    Number of save points in an interval
signal: (N) list
    Input signal
delMin: int
    Minimum delay (samples)
delMax: int
    Maximum delay (samples)

Returns
-------
result: {'durations' : durOut, 'result' : result} dict
    Autocorrelation vs delta at different save point intervals

Note
----
This function can be used to determine the equilibrium time, as it
is the autocorrelation defined in the same way as the util/auto_correlation
in the code base, evaluated at different save points.
'''
# </editor-fold>
def autoCorrCBVt(duration: int, signal: list, delMin: int, delMax: int):
    if delMax >= duration:
        raise ValueError('delta exceeds duration')
    numDurations = signal.shape[0] // duration
    durOut = np.array([[]])
    result = np.array([[]])
    for i in range(numDurations):
        thisSignal = signal[i * duration : (i + 1) * duration]
        autoCorrThisDur = autoCorrCB(thisSignal[:, 1], delMin, delMax)
        autoCorrThisDur = np.array([autoCorrThisDur])
        # Keep track of the duration of each rLAtn distribution
        thisDur = np.array([signal[i * duration, 0]])
        thisDur = np.array([thisDur])

        if i == 0:
            result = np.array(autoCorrThisDur)
            durOut = np.array(thisDur)
        else:
            result = np.append(result, autoCorrThisDur, axis=0)
            durOut = np.append(durOut, thisDur, axis=0)

    return {'durations' : durOut, 'result' : result}

# ---------- energy functions ----------#
# <editor-fold getEnergiesCol
''' Get energy parameters in energiesv0 at all save points

Parameters
----------
columns: list
    list of numbers corresponding to the columns in energiesv0 file

Returns
-------
result: (N, columns.shape[0] + 1) np.ndarary
    First column: save point index (starting from 0)
    Other columns: data corresponding to the column number

Note
----
See the columns corresponding to each energy at the module documentation
'''
# </editor-fold>
def getEnergiesCol(columns: list, directory: str):
    data = np.loadtxt(directory + 'energiesv0', skiprows=1)
    columns = np.append(0, columns)
    result = np.array([data[:, i] for i in columns])
    return np.transpose(result)

# <editor-fold getEnergies
''' Plot energies

Parameters
----------
energies: list (str)
    list of types of energies to plot
total: bool
    whether to include the sum of all energies specified in energies variable

Return
------
result: (N, number of energies + 1) np.ndarray
    First column is the save point.
    Next columns are energies corresponding to the save points.
    The energies included depend on the energies list; however, the ordering
    is always bend, stretch, shear, twist

Note
----
Energies available for this function are bend, stretch, shear, and twist.
'''
# </editor-fold>
def getEnergies(energies: list, directory: str, total: bool=False):
    energiesColumn = np.array([], dtype=int)
    if 'bend' in energies:
        energiesColumn = np.append(energiesColumn, 17)
    if 'stretch' in energies:
        energiesColumn = np.append(energiesColumn, 20)
    if 'shear' in energies:
        energiesColumn = np.append(energiesColumn, 23)
    if 'twist' in energies:
        energiesColumn = np.append(energiesColumn, 35)

    result = getEnergiesCol(energiesColumn, directory)
    if total:
        totalEnergy = np.sum(result[:, 1:], axis=1)
        result = np.concatenate((result.T, [totalEnergy]), axis=0)
        result = result.T

    return result

# ---------- end-to-end distance squared functions ----------#

# <editor-fold r2vt doc
''' Squared distances of beads n links apart versus save points

Parameters
----------
n : int
    Number of links between the two beads

Returns
-------
result: (N, 2) np.ndarray
    First column is the save point number.
    Second column is the squared distance at the corresponding save point.

Notes
-----
The first bead in the calculation is always the first bead of the chain.
Only one squared distance is calculated from each save point.
'''
# </editor-fold>
def r2vt(n: int, savePoints: list, directory: str):
    numSavePoints = (savePoints[1] - savePoints[0]) // savePoints[2] + 1
    result = np.empty([numSavePoints, 2])
    for i in range(numSavePoints):
        result[i, 0] = i * savePoints[2] + savePoints[0]
        fname = directory + 'r' + str(i * savePoints[2]+ savePoints[0]) + 'v0'
        thisR = np.loadtxt(fname)
        result[i, 1] = sum((thisR[n, :] - thisR[0, :]) ** 2)
    return result

# <editor-fold r2Atn doc
''' Squared distance of beads n links apart

Parameters
----------
n : int
    Number of links between the two beads

Returns
-------
result: (N) np.ndarray
    Squared distances of beads n links apart, compiled across different segments
    of the chain and different save points
'''
# </editor-fold>
def r2Atn(n: int, savePoints: list, directory: str):
    numBeads = getNumBeads(directory)
    numSegPerChain = (numBeads - 1) // n
    numSavePoints = (savePoints[1] - savePoints[0]) // savePoints[2] + 1

    result = np.array([])
    for i in range(numSavePoints):
        fname = directory + 'r' + str(i * savePoints[2]+ savePoints[0]) + 'v0'
        thisR = np.loadtxt(fname)
        for j in range(numSegPerChain):
            result = np.append(result, sum((thisR[(j + 1) * n, :] - thisR[j * n, :]) ** 2))
    return result

# <editor-fold r2Vn doc
''' Mean squared distance of beads n links apart versus n

Parameters
----------
nRange : (3) list
    Indicate values of n, the number of links between beads, to be used in calculation.
    The format is [nStart, nEnd, nNext], where
    nStart (int) is the starting n.
    nEnd (int) is the last n (inclusive).
    nNext (int) is the increment to the next n.

Returns
-------
result: (N, 2) np.ndarray
    First column is n, the number of links between beads.
    Second column is the mean squared distance.

Note
----
The squared distance is averaged across savePoints.
'''
# </editor-fold>
def r2Vn(nRange: list, savePoints: list, directory: str):
    numDataPoints = (nRange[1] - nRange[0]) // nRange[2] + 1
    result = np.empty([numDataPoints, 2])
    for i in range(numDataPoints):
        result[i, 0] = i * nRange[2] + nRange[0]
        result[i, 1] = np.mean(r2Atn(i * nRange[2] + nRange[0], savePoints, directory))
    return result

# ---------- relative end-to-end distance functions ----------#

# <editor-fold rLAtn doc
''' Distance of beads n links apart relative to maximum length of the chain

Parameters
----------
n : int
    Number of links between the two beads
l0: float
    Maximum length of the chain

Returns
-------
result: (N) np.ndarray
    Relative distances of beads n links apart, compiled across different segments
    of the chain and different save points

Note
----
The distance is the same as L2 norm of end-to-end vector.
'''
# </editor-fold>
def rLAtn(n: int, savePoints: list, l0: float, directory: str):
    numBeads = getNumBeads(directory)
    chainLength = l0 * n
    numSegPerChain = (numBeads - 1) // n
    numSavePoints = (savePoints[1] - savePoints[0]) // savePoints[2] + 1

    result = np.array([])
    for i in range(numSavePoints):
        fname = directory + 'r' + str(i * savePoints[2]+ savePoints[0]) + 'v0'
        thisR = np.loadtxt(fname)
        for j in range(numSegPerChain):
            result = np.append(result, np.sqrt(sum((thisR[(j + 1) * n, :] - thisR[j * n, :]) ** 2)) / chainLength)
    return result

# <editor-fold rLAtnVt doc
''' Distances of beads n links apart relative to maximum length of the chain
    at different intervals of save points

Parameters
----------
n : int
    Number of links between the two beads
duration: int
    Number of save points in an interval
l0: float
    Maximum length of the chain

Returns
-------
result : {'durations' : durOut, 'result' : result} dict
    Relative distances at different save point intervals

Note
----
The distance is the same as L2 norm of end-to-end vector.
'''
# </editor-fold>
def rLAtnVt(n: int, savePoints: list, duration: int, l0: float, directory: str):
    numDurations = (savePoints[1] - savePoints[0]) // duration

    durOut = np.array([[]])
    result = np.array([[]])
    for i in range(numDurations):
        rLAtThisDur = rLAtn(n, (i * duration + savePoints[0], (i + 1) * duration + savePoints[0], savePoints[2]), l0, directory)
        rLAtThisDur = np.array([rLAtThisDur])
        # Keep track of the duration of each rLAtn distribution
        thisDur = np.array([i * duration + savePoints[0], (i + 1) * duration + savePoints[0]])
        thisDur = np.array([thisDur])

        if i == 0:
            result = np.array(rLAtThisDur)
            durOut = np.array(thisDur)
        else:
            result = np.append(result, rLAtThisDur, axis=0)
            durOut = np.append(durOut, thisDur, axis=0)

    return {'durations' : durOut, 'result' : result}

# <editor-fold rLAtnVHist doc
''' Plot histograms of relative distances at different save point intervals

Parameters
----------
result : {'durations' : durOut, 'result' : result} dict
    Relative distances at different save point intervals, returns of rLAtnVt
numBins: int
    Number of bins for each histogram

Note
----
The distance is the same as L2 norm of end-to-end vector.
'''
# </editor-fold>
def rLAtnVtHist(result: dict, numBins: int):
    numDurations = result['result'].shape[0]
    for i in range(numDurations):
        plt.hist(result['result'][i], numBins, density=True, histtype='step', label='save points' + str(result['durations'][i]))
    plt.xlabel('R/L')
    plt.ylabel('P(R/L)')
    plt.title('End-to-end distance distribution')
    plt.legend()

# ---------- tangent correlator functions ----------#

# <editor-fold uuAtn doc
''' Tangent correlator of beads n links apart

Parameters
----------
n : int
    Number of links between the two beads

Returns
-------
result: (N) np.ndarray
    Tangent correlator of beads n links apart, compiled across different segments
    of the chain and different save points
'''
# </editor-fold>
def uuAtn(n: int, savePoints: list, directory: str):
    numBeads = getNumBeads(directory)
    numSegPerChain = (numBeads - 1) // n
    numSavePoints = (savePoints[1] - savePoints[0]) // savePoints[2] + 1

    result = np.array([])
    for i in range(numSavePoints):
        fname = directory + 'u' + str(i * savePoints[2]+ savePoints[0]) + 'v0'
        thisR = np.loadtxt(fname)
        for j in range(numSegPerChain):
            result = np.append(result, np.dot(thisR[(j + 1) * n, :], thisR[j * n, :]))
    return result

# <editor-fold uuVn doc
''' Tangent correlator of beads n links apart versus n

Parameters
----------
nRange : (3) list
    Indicate values of n, the number of links between beads, to be used in calculation.
    The format is [nStart, nEnd, nNext], where
    nStart (int) is the starting n.
    nEnd (int) is the last n (inclusive).
    nNext (int) is the increment to the next n.

Returns
-------
result: (N, 2) np.ndarray
    First column is n, the number of links between beads.
    Second column is the mean tangent correlator.

Note
----
The tangent correlator is averaged across savePoints.
'''
# </editor-fold>
def uuVn(nRange: list, savePoints: list, directory: str):
    numDataPoints = (nRange[1] - nRange[0]) // nRange[2] + 1
    result = np.empty([numDataPoints, 2])
    for i in range(numDataPoints):
        result[i, 0] = i * nRange[2] + nRange[0]
        result[i, 1] = np.mean(uuAtn(i * nRange[2] + nRange[0], savePoints, directory))
    return result

# ---------- radius of gyration functions ----------#
# <editor-fold rgVt doc
''' Radius of gyration versus save point

Returns
-------
result: (N, 2) np.ndarray
    First column is n, the number of links between beads.
    Second column is the radius of gyration.

Note
----
Radius of gyration is calculated as follows:
Rg = sqrt((1/N) * ||sum(Ri - Rcm)^2||)
where Rcm = (1/N) * sum(Ri)
'''
# </editor-fold>
def rgVt(savePoints: list, directory: str):
    numBeads = getNumBeads(directory)
    numSavePoints = (savePoints[1] - savePoints[0]) // savePoints[2] + 1
    result = np.empty([numSavePoints, 2])
    for i in range(numSavePoints):
        result[i, 0] = i * savePoints[2] + savePoints[0]
        fname = directory + 'r' + str(i * savePoints[2]+ savePoints[0]) + 'v0'
        thisR = np.loadtxt(fname)
        thisrCM = np.sum(thisR, axis=0) / numBeads
        result[i, 1] = np.sqrt(LA.norm(np.sum((thisR - thisrCM) ** 2, axis=0)) / numBeads)
    return result

# ---------- ring area functions ----------#

# <editor-fold areaRing
''' Area of the closed curve formed by a polymer ring

Parameters
----------
x: (N) list
    x positions of the beads
y: (N) list
    y positions of the beads
inteType: str
    Type of numerical integration used
    simps means Simpson's rule
    trapz means Trapezoidal rule

Returns
-------
Area: float
    Area of the projection of the ring on xy plane

Note
----
This function calculates the area of the projection on xy-plane of the ring
polymer by using Stokes theorem. Let the field F = k, where k is the unit vector
in z direction. Then, it can be shown that
    Area(projection on xy plane) = integral(C) xdy
where the right-hand side is the line integral of x with respect to y along the ring.
Note that the area can be negative.
'''
# </editor-fold>
def areaRing(x: list, y: list, inteType: str='trapz'):
    # Need to iterate back to the first point
    x_append = np.append(x, x[0])
    y_append = np.append(y, y[0])
    if inteType == 'simps':
        return inte.simps(x_append, y_append)
    elif inteType == 'trapz':
        return inte.trapz(x_append, y_append)
    else:
        raise ValueError('invalid inteType')

# <editor-fold areaRingCvH
''' Area of the closed curve formed by the convex hull a polymer ring

Parameters
----------
x: (N) list
    x positions of the beads
y: (N) list
    y positions of the beads
inteType: str
    Type of numerical integration used
    simps means Simpson's rule
    trapz means Trapezoidal rule

Returns
-------
Area: float
    Area of the convex hull of the projection of the ring on xy plane

Note
----
This function finds the convex hull of the projection of all points onto
xy plane. Then, calculate the area formed by that convex hull using areaRing
function.
'''
# </editor-fold>
def areaRingCvH(x: list, y: list, inteType: str='trapz'):
    xHull, yHull = positionCvH(x, y)
    return areaRing(xHull, yHull, inteType)

# <editor-fold positionCvH
''' Position of all points in the convex hull forming a curve

Parameters
----------
x: (N) list
    x positions of the beads
y: (N) list
    y positions of the beads

Returns
-------
xHull: (N) np.ndarray
    x coordinate of points in the convex hull in an order which forms a curve
yHull: (N) np.ndarray
    y coordinate of points in the convex hull in an order which forms a curve

Note
----
Note that ConvexHull returns a collection of line segments, specified by pairs
of points. However, the pairs are not in any particular order. indexCvH rearranges
these pairs of indices into a sequence of points along the "outer" curve formed
by the convex hull.
'''
# </editor-fold>
def positionCvH(x: list, y: list):
    points = np.transpose(np.concatenate(([x], [y])))
    hull = ConvexHull(points)

    # Indices of points forming a curve formed by convex hull
    hullIndices = curveCvH(hull.simplices)

    xHull = np.array([x[i] for i in hullIndices])
    yHull = np.array([y[i] for i in hullIndices])
    return xHull, yHull

# <editor-fold curveCvH
''' Position of all points in the convex hull forming a curve

Parameters
----------
simplices: (N, 2) list
    simplices of the convex hull. This can be obtained from hull.simplices
    when hull is the return of ConvexHull

Returns
-------
result: (N) np.ndarray
    indices of simplices of convex hull in a sequence that forms an "outer" curve

Note
----
This function rearranges the simplex pairs into a sequence of points along
the "outer" curve formed by the convex hull, so that area calculation is possible.
'''
# </editor-fold>
def curveCvH(simplices: list):
    result = np.array([], dtype=int);

    # Add first point
    result = np.append(result, simplices[0, 0])

    # Next point is simplices[nextIndex[0], nextIndex[1]]
    nextIndex = np.array([0, 1])

    for i in range(simplices.shape[0] - 1):
        result = np.append(result, simplices[nextIndex[0], nextIndex[1]])
        nextIndex = nextIndexCvH(simplices, nextIndex)

    return result

# <editor-fold nextIndexCvH
''' Index of the next adjecent simplex

Parameters
----------
simplices: (N, 2) list
    Simplices of the convex hull. This can be obtained from hull.simplices
    when hull is the return of ConvexHull
thisIndex: int
    Index of the current simplex

Returns
-------
result: (2) np.ndarray
    Index in simplices of the next adjecent point

Note
----
This function is the helper of curveCvH.
'''
# </editor-fold>
def nextIndexCvH(simplices: list, thisIndex: int):
    thisPoint = simplices[thisIndex[0], thisIndex[1]]
    # Search every point in simplices
    for i in range(simplices.shape[0]):
        for j in range(simplices.shape[1]):
            if (simplices[i, j] == thisPoint and i != thisIndex[0]):
                if j == 0:
                    jOfNext = 1
                else:
                    jOfNext = 0
                return np.array([i, jOfNext])

# <editor-fold areaRingVt
''' Area of the closed curve formed by a polymer ring versus save point

Parameters
----------
inteType: str
    Type of numerical integration used
    simps means Simpson's rule
    trapz means Trapezoidal rule
noNeg: bool
    If noNeg is true, returns the absolute value of the area.

Returns
-------
result: (N, 2) np.ndarray
    First column is the save point
    Second column is the corresponding area

Note
----
This function calculates the area of the projection on xy-plane of the ring
polymer by using Stokes theorem. Note that the area can be negative.
'''
# </editor-fold>
def areaRingVt(savePoints: list, directory: str, inteType: str='trapz', noNeg: bool=True):
    numSavePoints = (savePoints[1] - savePoints[0]) // savePoints[2] + 1
    result = np.empty([numSavePoints, 2])
    for i in range(numSavePoints):
        result[i, 0] = i * savePoints[2] + savePoints[0]
        fname = directory + 'r' + str(i * savePoints[2]+ savePoints[0]) + 'v0'
        thisR = np.loadtxt(fname)
        if noNeg:
            result[i, 1] = abs(areaRing(thisR[:, 0], thisR[:, 1], inteType))
        else:
            result[i, 1] = areaRing(thisR[:, 0], thisR[:, 1], inteType)
    return result

# <editor-fold areaRingCvHVt
''' Area of the closed curve formed by the convex hull of a polymer ring versus save point

Parameters
----------
inteType: str
    Type of numerical integration used
    simps means Simpson's rule
    trapz means Trapezoidal rule
noNeg: bool
    If noNeg is true, returns the absolute value of the area.

Returns
-------
result: (N, 2) np.ndarray
    First column is the save point
    Second column is the corresponding area

Note
----
This function calculates the area of the projection on xy-plane of the ring
polymer by using Stokes theorem. Note that the area can be negative.
'''
# </editor-fold>
def areaRingCvHVt(savePoints: list, directory: str, inteType: str='trapz', noNeg: bool=True):
    numSavePoints = (savePoints[1] - savePoints[0]) // savePoints[2] + 1
    result = np.empty([numSavePoints, 2])
    for i in range(numSavePoints):
        result[i, 0] = i * savePoints[2] + savePoints[0]
        fname = directory + 'r' + str(i * savePoints[2]+ savePoints[0]) + 'v0'
        thisR = np.loadtxt(fname)
        if noNeg:
            result[i, 1] = abs(areaRingCvH(thisR[:, 0], thisR[:, 1], inteType))
        else:
            result[i, 1] = areaRingCvH(thisR[:, 0], thisR[:, 1], inteType)
    return result
