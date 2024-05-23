class LIF_Behavior(Behavior) :

    def initialize(self, ng) : 
        super().initialize(ng)

        self.threshold = self.parameter("threshold", -10)
        self.reset = self.parameter("reset", -80)
        self.rest = self.parameter("rest", -65)
        self.tau = self.parameter("tau", 10)
        self.dt = self.parameter("dt", 1)
        self.R = self.parameter("R", 10)

        ng.voltage = ng.vector("normal(-60, 40)")

        firing = ng.voltage >= self.threshold
        ng.spike = firing.byte()
        ng.voltage[firing] = self.reset

        ng.activity = ng.vector(torch.sum(ng.spike).item()) / ng.size


    def forward(self, ng) : 
        firing = ng.voltage >= self.threshold
        ng.spike = firing.byte()
        ng.voltage[firing] = self.reset

        dV = (-(ng.voltage - self.rest) + self.R * ng.I) / self.tau 
        ng.voltage += dV * self.dt

        ng.activity = ng.vector(torch.sum(ng.spike).item()) / ng.size


class ELIF_Behavior(Behavior) : 

    def initialize(self, neuron) : 
        super().initialize(neuron)

        self.threshold = self.parameter("threshold", 30)
        neuron.reset = self.parameter("reset", -80)
        neuron.rest = self.parameter("rest", -65)
        neuron.tau = self.parameter("tau", 10)
        neuron.dt = self.parameter("dt", 0.1)
        neuron.R = self.parameter("R", 10)

        neuron.voltage = neuron.vector("ones") * neuron.rest

        # Exponential parameters

        self.rh_threshold = self.parameter("rh_threshold", -10)
        neuron.delta = self.parameter("delta", 5)
        
        neuron.spike_zone = False

    def forward(self, neuron) : 
        firing = torch.BoolTensor([(neuron.voltage >= self.rh_threshold and not neuron.spike_zone)])
            
        if(firing) : 
            neuron.spike_zone = True
        elif(not firing and neuron.voltage < self.rh_threshold) :
            neuron.spike_zone = False
        
        neuron.spike = firing.byte()
            
        reset_need = neuron.voltage >= self.threshold
        neuron.voltage[reset_need] = neuron.reset

        dV = (
            -(neuron.voltage - neuron.rest) * (not neuron.spike_zone)
            + neuron.delta*torch.exp(torch.tensor([(neuron.voltage - self.rh_threshold) / neuron.delta]))
            + neuron.R*neuron.I 
            ) / neuron.tau
        
        neuron.voltage += dV * neuron.dt
        neuron.voltage = torch.minimum(neuron.voltage, neuron.vector(100))

    
class AELIF_Behavior(Behavior) : 

    def initialize(self, neuron) : 
        super().initialize(neuron)

        # main LIF
        # -----------------------------------------------------------

        self.threshold = self.parameter("threshold", 30)
        neuron.reset = self.parameter("reset", -80)
        neuron.rest = self.parameter("rest", -65)
        neuron.tau = self.parameter("tau", 10)
        neuron.dt = self.parameter("dt", 1)
        neuron.R = self.parameter("R", 10)

        neuron.voltage = neuron.vector("normal(-65, 15)")

        # Exponential parameters

        self.rh_threshold = self.parameter("rh_threshold", -10)
        neuron.delta = self.parameter("delta", 5)

        neuron.spike_zone = False

        # Adaptability
        # -----------------------------------------------------------
        
        neuron.a = self.parameter("a", 0.2) # sub-threshold adaptation variable
        neuron.b = self.parameter("b", 0.7) # spike-triggered adaptation variable

        neuron.w = 0 # adaptive variable
        neuron.adpt_tau = self.parameter("adpt_tau", 10) # time-constant for w

    def forward(self, neuron) : 
        firing = torch.BoolTensor([(neuron.voltage >= self.rh_threshold and not neuron.spike_zone)])
            
        if(firing) : 
            neuron.spike_zone = True
        elif(not firing and neuron.voltage < self.rh_threshold) :
            neuron.spike_zone = False
        
        neuron.spike = firing.byte()
            
        reset_need = neuron.voltage >= self.threshold
        neuron.voltage[reset_need] = neuron.reset

        dV = (
            -(neuron.voltage - neuron.rest) * (not neuron.spike_zone)
            + neuron.delta*torch.exp(torch.tensor([(neuron.voltage - self.rh_threshold) / neuron.delta]))
            + neuron.R*neuron.I 
            - neuron.R*neuron.w * (not neuron.spike_zone)
        ) / neuron.tau
        
        neuron.voltage += dV * neuron.dt

        dW = (
            + neuron.a*(neuron.voltage - neuron.rest)
            - neuron.w
            + neuron.b*neuron.adpt_tau*(firing.byte())
        ) / neuron.adpt_tau

        neuron.w += dW * neuron.dt


class Adaptive_RAELIF(Behavior) :
    
    def initialize(self, neuron) : 
        super().initialize(neuron)

        # main LIF
        # -----------------------------------------------------------
        neuron.threshold = self.parameter("threshold", 30)
        neuron.reset = self.parameter("reset", -120)
        neuron.rest = self.parameter("rest", -65)
        neuron.tau = self.parameter("tau", 10)
        neuron.dt = self.parameter("dt", 1)
        neuron.R = self.parameter("R", 10)

        neuron.voltage = neuron.vector("normal(-40, 40)")

        # Exponential parameters

        neuron.rh_threshold = neuron.vector("ones") * self.parameter("rh_threshold", -10)
        neuron.delta = self.parameter("delta", 1)

        neuron.spike_zone = neuron.vector("zeros")

        # Adaptability
        # -----------------------------------------------------------
        
        self.a = self.parameter("a", 0.1) # sub-threshold adaptation variable
        self.b = self.parameter("b", 0.7) # spike-triggered adaptation variable

        neuron.w = neuron.vector("zeros") # adaptive variable
        self.adpt_tau = self.parameter("adpt_tau", 30) # time-constant for w

        # Refractory parameters
        # -----------------------------------------------------------

        neuron.T = neuron.vector("zeros") # number of iterations for refractory period
        neuron.refractory_period = neuron.vector("zeros")
        neuron.refractory_iter = self.parameter("refractory_iter", 20)

        # Adaptive Threshold
        # -----------------------------------------------------------

        neuron.tau_thresh_adpt = self.parameter("tau_thresh_adpt", 100) # adaptive threshold time constant
        neuron.eps = self.parameter("eps", 5) # epsilon changes of threshold when spiking
        neuron.base_threshold = -10


        # Initial Firing
        # -----------------------------------------------------------

        firing = torch.logical_and((neuron.voltage >= neuron.rh_threshold), torch.logical_not(neuron.spike_zone))
        deactive = torch.logical_and((neuron.voltage < neuron.rh_threshold), torch.logical_not(firing))
        
        neuron.spike_zone[firing] = True
        neuron.spike_zone[deactive] = False
        
        neuron.spike = firing.byte()
            
        reset_need = neuron.voltage >= neuron.threshold
        neuron.voltage[reset_need] = neuron.reset

        neuron.T[reset_need] = neuron.refractory_iter
        neuron.refractory_period[reset_need] = True

        neuron.T = neuron.T - 1
        neuron.T[neuron.T < 0] = 0
        
        neuron.refractory_period[neuron.T == 0] = False


    def forward(self, neuron) : 
        firing = torch.logical_and((neuron.voltage >= neuron.rh_threshold), torch.logical_not(neuron.spike_zone))
        deactive = torch.logical_and((neuron.voltage < neuron.rh_threshold), torch.logical_not(firing))
        
        neuron.spike_zone[firing] = True
        neuron.spike_zone[deactive] = False
        
        neuron.spike = firing.byte()
            
        reset_need = neuron.voltage >= neuron.threshold
        neuron.voltage[reset_need] = neuron.reset

        neuron.T[reset_need] = neuron.refractory_iter
        neuron.refractory_period[reset_need] = True

        neuron.T = neuron.T - 1
        neuron.T[neuron.T < 0] = 0
        
        neuron.refractory_period[neuron.T == 0] = False

        strike = torch.logical_not(torch.logical_and(neuron.spike_zone, (neuron.I < 0)))

        dV = (
            -(neuron.voltage - neuron.rest) + (-(neuron.voltage - neuron.rest) * 20 * neuron.refractory_period)
            + neuron.delta * torch.exp((neuron.voltage - neuron.rh_threshold) / neuron.delta)
            + neuron.R*neuron.I * (torch.logical_not(neuron.refractory_period)) * strike # refractory_period
            - neuron.R*neuron.w * (torch.logical_not(neuron.spike_zone))
            ) / neuron.tau
        
        neuron.voltage += dV * neuron.dt
        neuron.voltage = torch.minimum(neuron.voltage, neuron.vector(100))
        
        dW = (
            + self.a * (neuron.voltage - neuron.rest)
            - neuron.w
            + self.b * self.adpt_tau*(neuron.spike)
        ) / self.adpt_tau

        neuron.w += dW * neuron.dt

        dTh = (
            - (neuron.rh_threshold - neuron.base_threshold)     
            + neuron.eps * neuron.spike
        ) / neuron.tau_thresh_adpt

        neuron.rh_threshold += dTh * neuron.dt