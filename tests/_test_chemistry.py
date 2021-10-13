import numpy as np
from petitRADTRANS import poor_mans_nonequ_chem as pm

COs_goal_in = 0.55 * np.ones(2)
FEHs_goal_in =  0. * np.ones(2)
temps_goal_in = 1000. * np.ones(2)
pressures_goal_in = 0.01 * np.ones(2)

for i in range(1000):
    print(i)
    test = pm.interpol_abundances(COs_goal_in,FEHs_goal_in,temps_goal_in,pressures_goal_in)

print(test)
