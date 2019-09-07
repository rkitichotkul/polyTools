'''ringMatrices
Save plots of matrices of ring polymers
'''
import polyTools as pol
import matplotlib.pyplot as plt

directory = '../run1Twist0/'
name = 'run1Twist0'
savePoints = [0, 1000, 1]
eqSavePoint = 400
binNum = 10

dirData = directory + 'data/'
dirSave = directory

# Plot energies vs save point
energies = pol.getEnergies(['bend', 'stretch', 'shear'], dirData, total=True)
plt.plot(energies[:, 0], energies[:, 1], label='bend energy')
plt.plot(energies[:, 0], energies[:, 2], label='stretch energy')
plt.plot(energies[:, 0], energies[:, 3], label='shear energy')
plt.plot(energies[:, 0], energies[:, 4], label='total energy')
plt.xlabel('Save point')
plt.ylabel('Energy')
plt.title('Energies vs save point of ' + name)
plt.savefig(dirSave + '/energies_' + name + '.png')
plt.clf()

# Plot radius of gyration vs save point
Rg = pol.rgVt(savePoints, dirData)
plt.plot(Rg[:, 0], Rg[:, 1])
plt.xlabel('Save point')
plt.ylabel('Radius of gyration')
plt.title('Radius of gyration vs save point of ' + name)
plt.savefig(dirSave + '/rgVt_' + name + '.png')
plt.clf()

# Plot histogram of Rg at equilibrium
RgEq = Rg[eqSavePoint:, 1]
plt.hist(RgEq, binNum, density=True)
plt.xlabel('Probability')
plt.ylabel('Radius of Gyration (nm)')
plt.title('Distribution of Rg of ' + name)
plt.savefig(dirSave + '/rghist' + name + '.png')

# Plot area of projection on xy plane
area = pol.areaRingVt(savePoints, dirData, inteType='trapz')
plt.plot(area[:, 0], area[:, 1])
plt.xlabel('Save point')
plt.ylabel('Area')
plt.title('Area of projection on xy plane vs save point of ' + name)
plt.savefig(dirSave + '/areaRingVt_' + name + '.png')
plt.clf()

# Plot area of convex hull of projection on xy plane
area = pol.areaRingCvHVt(savePoints, dirData, inteType='trapz')
plt.plot(area[:, 0], area[:, 1])
plt.xlabel('Save point')
plt.ylabel('Area')
plt.title('Area of projection of convex hull on xy plane vs save point of ' + name)
plt.savefig(dirSave + '/areaRingCvHVt_' + name + '.png')
plt.clf()
