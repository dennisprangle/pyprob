import pyprob
import numpy as np
import ot
import torch
import cProfile

from pyprob.dis import ModelDIS
from showerSim import invMass_ginkgo
from torch.utils.data import DataLoader
from pyprob.nn.dataset import OnlineDataset
from pyprob.util import InferenceEngine
from pyprob.util import to_tensor
from pyprob import Model
from pyprob.model import Parallel_Generator
import math
from pyprob.distributions import Normal
from pyprob.distributions.delta import Delta


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as mpl_cm
plt.ion()

import sklearn as skl
from sklearn.linear_model import LinearRegression

from geomloss import SamplesLoss
sinkhorn = SamplesLoss(loss="sinkhorn", p=1, blur=.05)
def sinkhorn_t(x,y):
    x = to_tensor(x)
    y = torch.stack(y)
    return sinkhorn(x,y)

def ot_dist(x,y):
    # x = to_tensor(x)
    # y = torch.stack(y)
    x = np.array(x)
    y = np.array(torch.stack(y))
    a = ot.unif(len(x))
    b = ot.unif(len(y))
    Mat = ot.dist(x, y, metric='euclidean')
    #Mat1 /= Mat1.max()
    distance = to_tensor(ot.emd2(a,b,Mat))
    return distance


device = "cpu"

from pyprob.util import set_device
set_device(device)

obs_leaves = to_tensor([[44.57652381, 26.16169856, 25.3945314 , 25.64598258],
                           [18.2146321 , 10.70465096, 10.43553391, 10.40449709],
                           [ 6.47106713,  4.0435395,  3.65545951,  3.48697568],
                           [ 8.43764314,  5.51040615,  4.60990593,  4.42270416],
                           [26.61664145, 16.55894826, 14.3357362 , 15.12215264],
                           [ 8.62925002,  3.37121204,  5.19699   ,  6.00480461],
                           [ 1.64291837,  0.74506775,  1.01003622,  1.05626017],
                           [ 0.75525072,  0.3051808 ,  0.45721085,  0.51760643],
                           [39.5749915 , 18.39638928, 24.24717939, 25.29349408],
                           [ 4.18355659,  2.11145474,  2.82071304,  2.25221316],
                           [ 0.82932922,  0.29842766,  0.5799056 ,  0.509021  ],
                           [ 3.00825023,  1.36339397,  1.99203677,  1.79428211],
                           [ 7.20024308,  4.03280868,  3.82379277,  4.57441754],
                           [ 2.09953618,  1.28473579,  1.03554351,  1.29769683],
                           [12.21401828,  6.76059035,  6.94920042,  7.42823701],
                           [ 6.91438054,  3.68417135,  3.83782514,  4.41656731],
                           [ 1.97218904,  1.01632927,  1.08008339,  1.27454585],
                           [ 8.58164301,  5.06157833,  4.79691164,  4.99553141],
                           [ 5.97809522,  3.26557958,  3.4253764 ,  3.64894791],
                           [ 5.22842301,  2.94437891,  3.10292633,  3.00551074],
                           [15.40023764,  9.10884407,  8.93836964,  8.61970667],
                           [ 1.96101346,  1.24996337,  1.06923988,  1.06743143],
                           [19.81054106, 11.90268453, 11.60989346, 10.76953856],
                           [18.79470876, 11.429855  , 10.8377334 , 10.25112761],
                           [25.74331932, 15.63430056, 14.83860792, 14.07189108],
                           [ 9.98357576,  6.10090721,  5.68664128,  5.48748692],
                           [12.34604239,  7.78770185,  6.76075998,  6.78498685],
                           [21.24998531, 12.95180254, 11.9511704 , 11.87319933],
                           [ 7.80693733,  4.83117128,  4.27443559,  4.39602348],
                           [16.28983576,  9.66683929,  9.24891886,  9.28970032],
                           [ 2.50706736,  1.53153206,  1.36060018,  1.43002765],
                           [ 3.73938645,  2.06006639,  2.31013974,  2.09378969],
                           [20.2174725 , 11.88622367, 12.05106468, 11.05325362],
                           [ 9.48660008,  5.53665456,  5.54171966,  5.34966654],
                           [ 2.65812987,  1.64102742,  1.67392209,  1.25083707]])


QCD_mass = to_tensor(30.)
#rate=to_tensor([QCD_rate,QCD_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet
jetdir = to_tensor([1.,1.,1.])
jetP = to_tensor(400.)
jetvec = jetP * jetdir / torch.linalg.norm(jetdir) ## Jetvec is 3-momentum. JetP is relativistic p.


# Actual parameters
pt_min = to_tensor(0.3**2)
M2start = to_tensor(QCD_mass**2)
jetM = torch.sqrt(M2start) ## Mass of initial jet
jet4vec = torch.cat((torch.sqrt(jetP**2 + jetM**2).reshape(-1), jetvec))
minLeaves = 1
maxLeaves = 10000 # unachievable, to prevent rejections
maxNTry = 100



class SimulatorModelDIS(invMass_ginkgo.SimulatorModel, ModelDIS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dummy_bernoulli(self, jet):
        return True

    def forward(self, inputs=None):
        assert inputs is None # Modify code if this ever not met?
        # Sample parameter of interest from Unif(0,10) prior
        root_rate = pyprob.sample(pyprob.distributions.Uniform(0.01, 10.),
                                  name="decay_rate_parameter")
        decay_rate = pyprob.sample(pyprob.distributions.Uniform(0.01, 10.),
                                   name="decay_rate_parameter")
        # Simulator code needs two decay rates for (1) root note (2) all others
        # For now both are set to the same value
        inputs = [root_rate, decay_rate]
        jet = super().forward(inputs)
        delta_val = self.dummy_bernoulli(jet)
        bool_func_dist = pyprob.distributions.Bernoulli(delta_val)
        pyprob.observe(bool_func_dist, name = "dummy")
        return jet

# Make instance of the simulator
simulatorginkgo = SimulatorModelDIS(jet_p=jet4vec,  # parent particle 4-vector
                                    pt_cut=float(pt_min),  # minimum pT for resulting jet
                                    Delta_0= M2start,  # parent particle mass squared -> needs tensor
                                    M_hard=jetM,  # parent particle mass
                                    minLeaves=1,  # minimum number of jet constituents
                                    maxLeaves=10000,  # maximum number of jet constituents (a large value to stop expensive simulator runs)
                                    suppress_output=True,
                                    obs_leaves=obs_leaves,
                                    dist_fun=sinkhorn_t)


# dataset = Parallel_Generator(simulatorginkgo, importance_sample_size=1000, observe ={'dummy':1})
# dataloader = DataLoader(dataset, num_workers=12,batch_size= None)
#
# for (i,j) in enumerate(dataloader):
#     if (i+1)%100 == 0:
#         print(i+1)



#torch.multiprocessing.set_sharing_strategy("file_system")

traces = simulatorginkgo._dis_traces(num_traces = 1000, observe = {'dummy':1}, num_workers = 4, batch_size = 20)
#print(traces[0])



# if __name__ == '__main__':
#     simulatorginkgo.train(iterations=2, importance_sample_size=5000)
#     simulatorginkgo.save_inference_network('ginkgo_fixed_network_2')