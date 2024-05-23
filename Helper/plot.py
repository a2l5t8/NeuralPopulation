def plot_V(ng, cnt = 10) : 
    plt.plot(ng['voltage', 0][:,0:cnt])
    plt.plot(ng['torch.mean(voltage)', 0], color='black')
    plt.axhline(ng['LIF_Behavior', 0].threshold, color='black', linestyle='--')


def plot_I(ng, cnt = 10) : 
    plt.plot(ng['I', 0][:,0:cnt])
    plt.xlabel('iterations')
    plt.ylabel('input current')
    plt.title('Input Current of LIF model')
    plt.show()


def plot_spike_raster(ng) :
    plt.plot(ng['spike.t', 0], ng['spike.i', 0], '.', color = "blue")
    plt.xlabel('iterations')
    plt.ylabel('neuron index')
    plt.title('Raster Plot')
    plt.show()

def plot_activity(ng) : 
    plt.plot(ng['activity', 0][:,0:1])
    plt.xlabel('iterations')
    plt.ylabel('A(t)')
    plt.title('Population activity')
    plt.show()


def plot_all(ng) : 

    plotter.plot_V(ng)
    plotter.plot_I(ng)
    plotter.plot_spike_raster(ng)
    plotter.plot_activity(ng)