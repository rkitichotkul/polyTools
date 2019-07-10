
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl


# In[ ]:

# savePoints: list(start, end, skip) inclusive
def r2vt(n: int, savePoints: list, directory: str):
    numSavePoints = (savePoints[1] - savePoints[0]) // savePoints[2] + 1
    result = np.empty([numSavePoints, 2])
    for i in range(numSavePoints):
        result[i, 0] = i * savePoints[2] + savePoints[0]
        fname = directory + 'r' + str(i * savePoints[2]+ savePoints[0]) + 'v0'
        thisR = np.loadtxt(fname)
        result[i, 1] = sum((thisR[n, :] - thisR[0, :]) ** 2)
    return result


# In[ ]:


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


# In[ ]:


def r2Vn(nRange: list, savePoints: list, directory: str):
    numDataPoints = (nRange[1] - nRange[0]) // nRange[2] + 1
    result = np.empty([numDataPoints, 2])
    for i in range(numDataPoints):
        result[i, 0] = i * nRange[2] + nRange[0]
        result[i, 1] = np.mean(r2Atn(i * nRange[2] + nRange[0], savePoints, directory))
    return result


# In[ ]:


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


# In[49]:


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


# In[ ]:


def rLAtnVtHist(result: dict, numBins: int, directory: str=''):
    numDurations = result['result'].shape[0]
    for i in range(numDurations):
        fig, ax = plt.subplots()
        ax.hist(result['result'][i], numBins, density=True)
        ax.set_xlabel('R/L')
        ax.set_ylabel('P(R/L)')
        ax.set_title('End-to-end distance distribution, Save points: ' + str(result['durations'][i]))
        if (directory != ''):
            fig.savefig(directory + str(result['durations'][i][1]))


# In[ ]:


def rLAtnVtHist2(result: dict, numBins: int):
    numDurations = result['result'].shape[0]
    for i in range(numDurations):
        plt.hist(result['result'][i], numBins, density=True, histtype='step', label='save points' + str(result['durations'][i]))
    plt.xlabel('R/L')
    plt.ylabel('P(R/L)')
    plt.title('End-to-end distance distribution')
    plt.legend()


# In[ ]:


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


# In[ ]:


def uuVn(nRange: list, savePoints: list, directory: str):
    numDataPoints = (nRange[1] - nRange[0]) // nRange[2] + 1
    result = np.empty([numDataPoints, 2])
    for i in range(numDataPoints):
        result[i, 0] = i * nRange[2] + nRange[0]
        result[i, 1] = np.mean(uuAtn(i * nRange[2] + nRange[0], savePoints, directory))
    return result


# In[ ]:

def getNumBeads(directory: str):
    fname = directory + 'r0v0'
    thisR = np.loadtxt(fname)
    return thisR.shape[0]


def getEnergies(columns: list, directory: str):
    data = np.loadtxt(directory + 'energiesv0', skiprows=1)
    columns = np.append(0, columns)
    result = np.array([data[:, i] for i in columns])
    return result
