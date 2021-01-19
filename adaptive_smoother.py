import pickle
import os
import numpy as np
import seaborn as sns
import sys
import time
sns.set(style = 'white', font_scale = 1.5)

import matplotlib.pyplot as plt
path = "."
with open(os.path.join(path, "US101_Lane1to5_t30s30.pickle"), "rb") as f:
    data = pickle.load(f)


# data
rho = data['rhoMat']
q = data['qMat']
u = data['vMat']
dx = data['s'][1] - data['s'][0]
dt = data['t'][1] - data['t'][0]

param = {"sigma": dx/2,
        "tau": dt/1, #1.2,
        "c_free": 0.277778*110, # km/h to m/s
        "c_cong": 0.277778*(-15),
        "V_thr": 0.277778*60,
        "DV": 0.277778*20,
         "dx":dx,
         "dt":dt
        }

LOOPS = {2: [0,2],\
    3: [0,7,20],\
    4: [0,2,17,20],\
    5: [0,4,8,13,20],\
    6: [0,3,6,15,18,20],\
    7: [0,3,6,9,12,15,20],\
    8: [0,2,5,8,11,13,16,20],\
    9: [0,2,4,7,9,12,14,17,20],\
    10: [0,2,4,6,8,11,13,15,17,20],\
    11: [0,2,4,6,8,10,12,14,16,18,20],\
    12: [0,1,3,5,7,9,11,12,14,16,18,20],\
    13: [0,1,3,5,6,8,10,11,13,15,16,18,20],\
    14: [0,1,3,4,6,7,9,11,12,14,15,17,18,20],\
    15: [0,1,2,4,5,7,8,10,11,13,14,16,17,19,20],\
    16: [0,1,2,4,5,6,8,9,11,12,13,15,16,17,19,20],\
    18: [0,1,2,3,4,6,7,8,9,11,12,13,14,15,17,18,19,20]}

# the time index of the observation, which is the whole index set
T = np.array([i for i in range(rho.shape[1])])

ON_LINE = False # whether to use the "future" observation
class AdaSM():
    def __init__(self, z, X, T, param):
        # z: input data, could be rho, q, or v, which is the known loop data
        # sigma, tau: kernel smoothing factor
        # X,T: idx of the known looper
        self.z = z
        self.X = X
        self.T = T
        self.sigma = param["sigma"]
        self.tau = param["tau"]
        self.c_free = param["c_free"]
        self.c_cong = param["c_cong"]
        self.V_thr = param['V_thr']
        self.DV = param["DV"]
        self.dx = param["dx"]
        self.dt = param["dt"]
    def _phi(self,x,t):
        return np.exp( -1*(abs(x)/self.sigma+abs(t)/self.tau)  )
    def _V_free(self,x,t):
        # x and t is the index instead of absolute value
        numerator = 0
        denominator = 0
        if ON_LINE is True:
            new_T = self.T[np.where(T<= t)]
        else:
            new_T = self.T
        for x_i in self.X:
            for t_i in new_T:
                dist_x = (x_i-x)*self.dx
                dist_t = (t*self.dt - (x-x_i)*self.dx/self.c_free) - t_i*self.dt
                numerator += self._phi(dist_x, dist_t)*self.z[x_i, t_i]
                denominator += self._phi(dist_x, dist_t)
        return numerator / denominator
    def _V_cong(self,x,t):
        numerator = 0
        denominator = 0
        if ON_LINE is True:
            new_T = self.T[np.where(T<= t)]
        else:
            new_T = self.T
        for x_i in self.X:
            for t_i in new_T:
                dist_x = (x_i-x)*self.dx
                dist_t = (t*self.dt - (x-x_i)*self.dx/self.c_cong) - t_i*self.dt
                numerator += self._phi(dist_x, dist_t)*self.z[x_i, t_i]
                denominator += self._phi(dist_x, dist_t)
        return numerator / denominator
    
    def _w(self,x,t):
        V_free = self._V_free(x,t)
        V_cong = self._V_cong(x,t)
        W = 0.5*( 1 + np.tanh( (self.V_thr - np.min([V_free,V_cong])/self.DV) ) )
        return W
    def calculate_z(self,x,t):
        Z = self._w(x,t)*self._V_cong(x,t) + (1-self._w(x,t))*self._V_free(x,t)
        return Z
    
def get_error(z, X, T, param):
    adasm = AdaSM(z, X, T, param)
    z_star = np.zeros(z.shape)
    for i in range(z_star.shape[0]):
        for j in range(z_star.shape[1]):
            z_star[i,j] = adasm.calculate_z(i,j)
    error_z = np.linalg.norm(z-z_star,2)/np.linalg.norm(z,2)
    return z_star, error_z

def main(num_looper):
    if num_looper in LOOPS.keys():
        X = LOOPS[num_looper]
        rho_star, error_rho = get_error(rho, X, T, param)
        print('# of loop detector = %d        error_rho = %.3e'%(num_looper, error_rho))
    else:
        print('invalid # of loop detector')
        print('valid # of loop detector includes: ' + str(list(LOOPS.keys())))
    
    

if __name__ == "__main__":
    start_time = time.time()
    print('----running-------')
    main(int(sys.argv[1]))
    end_time = time.time()
    print('run_time:', end_time - start_time)