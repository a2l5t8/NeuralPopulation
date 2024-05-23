net = Network()

k = 8
I_init = torch.randint(15, 22, (k,))

target = 2
I_init[target] = 24

ngs = []
for i in range(k) : 
    md = "constant"
    if(i == target) : 
        md = "change"

    ng = NeuronGroup(net = net, size = 80, tag = "EX{}".format(i + 1), behavior={
        1 : CurrentBehavior(mode = md, has_noise = True, pw = I_init[i].item()),
        4 : SynTypeInput(),
        # 5 : DecisionDynamic(),
        6 : LIF_Behavior(tau = 25),
        # 7 : DecisionLIF(),
        9 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity']),
        10 : EventRecorder(['spike'])
    })

    ngs.append(ng)

ng_in = NeuronGroup(net = net, size = k * 20, tag = "INH", behavior={
    1 : CurrentBehavior(mode = "constant", has_noise = False, pw = 0),
    4 : SynTypeInput(),
    # 5 : DecisionDynamic(gain = lambda x : 1.4 * x),
    6 : LIF_Behavior(tau = 25),
    # 7 : DecisionLIF(gain = lambda x : 4 * x),
    9 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity']),
    10 : EventRecorder(['spike'])
})

for i in range(k) : 
    SynapseGroup(net = net, src = ngs[i], dst = ngs[i], tag = "GLUTAMATE", behavior={
        2 : SynConnectivity(mode = "balanced_fixed", C = 30, J0 = 40),
        3 : SynFun()
    })

    SynapseGroup(net = net, src = ngs[i], dst = ng_in, tag = "GLUTAMATE", behavior={
        2 : SynConnectivity(mode = "balanced_fixed", C = 30, J0 = 20),
        3 : SynFun()
    })

    SynapseGroup(net = net, src = ng_in, dst = ngs[i], tag = "GABA", behavior={
        2 : SynConnectivity(mode = "balanced_fixed", C = 30, J0 = 25),
        3 : SynFun()
    })


net.initialize()
net.simulate_iterations(100)


fig, (ax1) = plt.subplots(1, 1)
fig.suptitle('Raster Plot population')
# fig.set_figwidth()

lgd = []
for i in range(k) : 
    ax1.plot(net.NeuronGroups[i]['spike.t', 0], net.NeuronGroups[i]['spike.i', 0] + (i * 80), '.')
    lgd.append(chr(ord('A') + i))

plt.legend(lgd)
plt.show()

for i in range(k) : 
    plt.plot(net.NeuronGroups[i]['activity', 0][:,0:1])

plt.legend(lgd)
plt.show()

for i in range(k) : 
    plt.plot(torch.sum(net.NeuronGroups[i]['I', 0][:], axis = 1)/net.NeuronGroups[i].size)

plt.legend(lgd)
plt.show()

I_st = []
for ng in net.NeuronGroups : 
    if("INH" not in ng.tags) : 
        I_st.append((torch.sum(ng['I', 0][:], axis = 1)/ng.size)[0])

I_en = []
for ng in net.NeuronGroups : 
    if("INH" not in ng.tags) : 
        I_en.append((torch.sum(ng['I', 0][:], axis = 1)/ng.size)[-1])

plt.bar(lgd, height=I_init, width=0.6)
plt.show()

plt.bar(lgd, height=I_en, width=0.6)
plt.show()

act_st = []
for ng in net.NeuronGroups : 
    if("INH" not in ng.tags) :
        act_st.append(ng['activity', 0][:,0:1][4].item())

act_en = []
for ng in net.NeuronGroups : 
    if("INH" not in ng.tags) : 
        act_en.append(ng['activity', 0][:,0:1][-1].item())

plt.bar(lgd, height=act_st, width=0.6)
plt.show()

plt.bar(lgd, height=act_en, width=0.6)
plt.show()