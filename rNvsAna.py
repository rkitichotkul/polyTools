import numpy as np
import matplotlib.pyplot as plt
import pickle

''' Return r/L = |r(n) - r(0)|/L = sqrt(r2)/L of selected segments, within the chain and across time steps.
L is the total length of the polymer. Always start at the first bead (0th) in the chain.
Args:
    n = number of links in a segment
    numBeads = number of beads in the polymer chain
    timeStart = time step at which we begin to evaluate r2 (suggestion: when equilibrium is reached)
    numSavePoints = number of total save points (time steps), excluding the first one (the initial time t = 0)
    L = total length of the polymer (arbitrary unit, same as the unit of l0, bead-bead length)
    directory: in '.../data/' format
    segSkip = number of links skipped when finding r2 of segments within a chain (bead: 0, segSkip, 2 * segSkip, ...)
Returns:
    np.array: Array of r/L values'''

def rRelAtn(n, numBeads, timeStart, numSavePoints, L, directory, segSkip=1):
    if n >= numBeads:
        raise ValueError('segments cannot be longer than the chain')

    # numDataPoints = numDataPointsPerSavePoint * num(effective)SavePoints
    numDPPerSP = int((numBeads - n)/segSkip) if ((numBeads - n)/segSkip) % 1 == 0 else int(np.floor((numBeads - n)/segSkip) + 1)
    numSP = numSavePoints - timeStart + 1
    numDataPoints = numDPPerSP * numSP
    # Optional: to keep track of the calculation
    # print(numDataPoints)

    result = np.zeros(numDataPoints)
    for i in range(numSP):
        fname = directory + 'r' + str(i + timeStart) + 'v0'
        r_ti = np.loadtxt(fname)
        for j in range(numDPPerSP):
            for k in range(3):
                result[i * numDPPerSP + j] += (r_ti[j * segSkip, k] - r_ti[j * segSkip + n, k]) ** 2
            result[i * numDPPerSP + j] = np.sqrt(result[i * numDPPerSP + j])/L
    return result

print('Functions loaded')

ana = pickle.load(open('/home/users/rk22/pyTools/analytical_endend', 'rb'))

simRept = {}
rN250Rept = rRelAtn(1000, 1001, 500, 3000, 10000, '/home/users/rk22/wlcsim/data/')
rN125Rept = rRelAtn(500, 1001, 500, 3000, 5000, '/home/users/rk22/wlcsim/data/')
rN62p5Rept = rRelAtn(250, 1001, 500, 3000, 2500, '/home/users/rk22/wlcsim/data/')
rN31p25Rept = rRelAtn(125, 1001, 500, 3000, 1250, '/home/users/rk22/wlcsim/data/')
rN6p25Rept = rRelAtn(25, 1001, 500, 3000, 250, '/home/users/rk22/wlcsim/data/')
rN1p25Rept = rRelAtn(5, 1001, 500, 3000, 50, '/home/users/rk22/wlcsim/data/')
rNp25Rept = rRelAtn(1, 1001, 500, 3000, 10, '/home/users/rk22/wlcsim/data/')
simRept['N=250.0'] = rN250Rept
simRept['N=125.0'] = rN125Rept
simRept['N=62.5'] = rN62p5Rept
simRept['N=31.25'] = rN31p25Rept
simRept['N=6.25'] = rN6p25Rept
simRept['N=1.25'] = rN1p25Rept
simRept['N=0.25'] = rNp25Rept
print('Load R/L for N = 250, 125, 62.5, 31.25, 6.25, 1.25, 0.25 from the simulation')

for i in simRept.keys():
    if i == 'N=250.0':
        plt.hist(simRept[i], 50, density=True)
    else:
        plt.hist(simRept[i], 200, density=True)

    plt.plot(ana[i]['rOverL'], ana[i]['P'])
    plt.xlabel('R/L')
    plt.ylabel('P(R/L)')
    plt.title('Distribution of end-to-end distance for ' + i + ' when twist energy is enabled')
    plt.savefig('/home/users/rk22/archived/noTwist/r' + i + '.png')
    plt.clf()
