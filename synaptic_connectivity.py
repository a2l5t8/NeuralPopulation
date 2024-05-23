from pymonntorch import *
import numpy as np
import torch
from matplotlib import pyplot as plt
import random
import math

from Helper import plot as plotter


net = Network()

ng = NeuronGroup(net = net, size = 100, behavior={
    1 : CurrentBehavior(mode = "constant", has_noise = True, pw = 10),
    3 : SynInp(),
    6 : LIF_Behavior(tau = 25),
    9 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity']),
    10 : EventRecorder(['spike'])
})

sg = SynapseGroup(net = net, src = ng, dst = ng, behavior={
    2 : SynConnectivity(
        mode = "random_fixed",
        C = 30,
        J0 = 40,
        p = 0.3
    ),
    3 : SynFun()
})

net.initialize()
net.simulate_iterations(100)

plotter.plot_all(ng)