class SynFun(Behavior) : 

    def initialize(self, sg) : 
        # sg.W = sg.matrix(mode="normal(0.5, 0.3)")
        sg.I = sg.dst.vector()

    def forward(self, sg) :
        sg.I = torch.sum(sg.W[sg.src.spike], axis = 0)


class SynInp(Behavior) : 
    
    def forward(self, ng) : 
        for syn in ng.afferent_synapses["All"] : 
            ng.I += syn.I


class SynVoltInp(Behavior) : 
    def forward(self, ng) : 
        ng.spike_train = ng.vector("zeros")
        for syn in ng.afferent_synapses["All"] : 
            ng.spike_train += syn.I


class SynVolt(Behavior) :

    def initialize(self, ng) : 
        self.E_ex = self.parameter("E_ex", -75)
        self.tau_syn = self.parameter("tau", 1.2)
        self.gE_bar = self.parameter("gE_bar", 0.03)

        ng.gt = sg.dst.vector("zeros")
        
        self.dt = 1

    def forward(self, ng) : 
        dG = -(ng.gt) / self.tau_syn + self.gE_bar * ng.spike_train
        ng.gt += dG * self.dt

        ng.I += ng.gt * (ng.voltage - self.E_ex)

class SynTypeInput(Behavior) : 

    def forward(self, ng) : 
        for syn in ng.afferent_synapses["GLUTAMATE"] : 
            ng.I += syn.I
        
        for syn in ng.afferent_synapses["GABA"] : 
            ng.I -= syn.I


class SynConnectivity(Behavior) : 
    
    def initialize(self, sg) : 
        self.mode = self.parameter("mode", "full")
        self.J0 = self.parameter("J0", 50)
        self.n = self.parameter("size", 100)
        self.p = self.parameter("p", 0.3)
        self.C = self.parameter("C", 20)
        
        if(self.mode == "full") : 
            sg.W = sg.matrix(self.J0 / self.n)

        elif(self.mode == "full2") : 
            sg.W = sg.matrix("normal(1, 0.5)")

        elif(self.mode == "inh") : 
            sg.W = sg.matrix("normal(9, 3)")
        
        elif(self.mode == "random") : 
            sg.W = sg.matrix(self.J0 / (self.n * self.p))
            for i in range(sg.matrix_dim()[0]) : 
                for j in range(sg.matrix_dim()[1]) : 
                    q = random.random()
                    if(q > self.p) : 
                        sg.W[i, j] = 0

        elif(self.mode == "balanced") : 
            sg.W = sg.matrix(self.J0 / math.sqrt((self.n * self.p)))
            for i in range(sg.matrix_dim()[0]) : 
                for j in range(sg.matrix_dim()[1]) : 
                    q = random.random()
                    if(q > self.p) : 
                        sg.W[i, j] = 0

        elif(self.mode == "random_fixed") :
            sg.W = sg.matrix(self.J0 / self.n)
            for i in range(sg.matrix_dim()[0]) :
                adj = torch.randperm(sg.matrix_dim()[1])
                for j in range(self.C, sg.matrix_dim()[1]) : 
                    sg.W[i, adj[j]] = 0

        elif(self.mode == "balanced_fixed") : 
            md = "normal({}, {})".format(self.J0 / math.sqrt(self.C), self.J0 / math.sqrt(self.C) / 2)
            sg.W = sg.matrix(md)
            for i in range(sg.matrix_dim()[0]) :
                adj = torch.randperm(sg.matrix_dim()[1])
                for j in range(self.C, sg.matrix_dim()[1]) : 
                    sg.W[i, adj[j]] = 0            