

from mpi4py import MPI
from scipy.stats import uniform
from particles import smc_samplers as ssp
# standard libraries
from matplotlib import pyplot as plt
import numpy as np
from time import time
import pickle
# modules from particles
import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class ToyModel(ssp.StaticModel):
    def __init__(self, base_dist, data):
      self.prior = base_dist
      self.data = data
    def logpyt(self, theta, t):  # density of Y_t given theta and Y_{0:t-1}
        return dists.MvNormal(loc = theta['mu'], cov = np.eye(d)).logpdf(self.data[t])

# Very simple model, we perform Bayesian inference to infer the value of the mean of our data.
# We have a prior of mean 0 and we want to find that the mean is 10.
d = 5
data = dists.MvNormal(loc = 10., cov=np.eye(d)).rvs(size = 100)
my_prior = dists.StructDist({'mu': dists.MvNormal(loc = 0., cov=np.eye(d))})


N = 100
len_chain = 10000

#comm.Barrier()
times = []
my_static_model = ToyModel(data = data, base_dist = my_prior)  
move = ssp.AdaptiveMCMCSequence(len_chain = len_chain)
if size >1 :
    fk_tempering = ssp.AdaptiveTempering(my_static_model, move = move, wastefree = False, mpi_comm = comm)
else:
    fk_tempering = ssp.AdaptiveTempering(my_static_model, move = move, wastefree = False)

if rank == 0:  
    if size >1:
        my_alg = particles.SMC(fk=fk_tempering, N=N, ESSrmin=1, verbose=True, store_history = True, mpi_comm = comm)
    else:
        my_alg = particles.SMC(fk=fk_tempering, N=N, ESSrmin=1, verbose=True, store_history = True)
    

n_it_max = 1000
n_it = 1000
time1 = time()
for t in range(1, n_it):
    if rank == 0:
        try:
            next(my_alg)
        except StopIteration:
            n_it_max = 0
            time_tot = time() - time1
            times.append(time_tot)
            print(time_tot, "for N = ", N, "and len", len_chain )
    else:
        req1 = comm.irecv(source=0, tag=123)
        req2 = comm.irecv(source=0, tag=124)

        dict1 = req1.wait()
        shared = req2.wait()
        dict1 = pickle.loads(dict1)
        
        scattered_inputs = ssp.ThetaParticles(theta = dict1['theta'], lprior = dict1['lprior'], llik = dict1['llik'], lpost = dict1['lpost'])
        scattered_inputs.shared['exponents'] = shared[0]
        scattered_inputs.shared['chol_cov'] = shared[1]
        
        local_result = fk_tempering.M(t, scattered_inputs)
        dict1 = pickle.dumps(local_result.dict_fields)
        
        req1 = comm.isend(dict1, dest=0, tag=234)
        req1.wait()
        
    
    n_it_max = comm.bcast(n_it_max, root=0)
    if n_it_max <= t:
        break
if rank == 0:
    result = np.mean((np.exp(my_alg.hist.wgts[-1].W)*(my_alg.hist.X[-1].theta['mu'].T)).T), np.var((np.exp(my_alg.hist.wgts[-1].W)*(my_alg.hist.X[-1].theta['mu'].T)).T)
    print("Résulat: la moyenne est approximée à ", result[0], "avec une variance de", result[1])    
