net = Network()

ng_ex = NeuronGroup(net = net, size = 80, behavior={
    1 : CurrentBehavior(mode = "constant", has_noise = False, pw = 10),
    4 : SynTypeInput(),
    6 : LIF_Behavior(tau = 25),
    9 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity']),
    10 : EventRecorder(['spike'])
})

ng_in = NeuronGroup(net = net, size = 20, behavior={
    1 : CurrentBehavior(mode = "constant", has_noise = False, pw = 10),
    4 : SynTypeInput(),
    6 : LIF_Behavior(tau = 22),
    9 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity']),
    10 : EventRecorder(['spike'])
})

SynapseGroup(net = net, src = ng_ex, dst = ng_ex, tag = "GLUTAMATE", behavior={
    2 : SynConnectivity(mode = "balanced_fixed", J0 = 30, C = 30),
    3 : SynFun()
})

SynapseGroup(net = net, src = ng_ex, dst = ng_in, tag = "GLUTAMATE", behavior={
    2 : SynConnectivity(mode = "balanced_fixed", J0 = 30, C = 30),
    3 : SynFun()
})

SynapseGroup(net = net, src = ng_in, dst = ng_ex, tag = "GABA", behavior={
    2 : SynConnectivity(mode = "balanced_fixed", J0 = 30, C = 30),
    3 : SynFun()
})

SynapseGroup(net = net, src = ng_in, dst = ng_in, tag = "GABA", behavior={
    2 : SynConnectivity(mode = "balanced_fixed", J0 = 30, C = 30),
    3 : SynFun()
})

net.initialize()
net.simulate_iterations(100)

plotter.plot_V(ng)
plotter.plot_I(ng)
plotter.plot_spike_raster(ng)
plotter.plot_activity(ng)