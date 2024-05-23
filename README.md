# Neural Population

Neural Population is a comprehensive project that explores the intricate dynamics of neural networks, focusing on various aspects such as synaptic connectivity schemes, balanced network design using excitatory and inhibitory neurons, and decision-making mechanisms through winner-takes-all and inhibitory populations. This repository provides a robust framework for simulating and analyzing these complex neural phenomena.

## Features

### 1. Implementation of Different Synaptic Connectivity Schemes
- **Random Connectivity**: Simulates networks where synapses are formed randomly between neurons.
- **Structured Connectivity**: Implements specific patterns of synaptic connections based on predefined rules or data.
- **Full Connectivity**: Simulating small population fully conncted to each other.

### 2. Designing a Balanced Network Using Excitatory and Inhibitory Neurons
- **Network Composition**: Configures networks with a mix of excitatory and inhibitory neurons to achieve balance.
- **Stability Analysis**: Provides tools to analyze the stability and robustness of the network.
- **Parameter Tuning**: Offers adjustable parameters for synaptic strengths, neuron types, and connectivity ratios to fine-tune network balance.

### 3. Dynamics of Decision Making Using Winner-Takes-All and Inhibitory Populations
- **Winner-Takes-All Mechanism**: Models decision-making processes where the most active neuron inhibits others, effectively "winning" the competition.
- **Inhibitory Population Dynamics**: Explores how groups of inhibitory neurons influence decision-making and network behavior.
- **Simulation Tools**: Includes scripts for simulating decision-making scenarios and visualizing the outcomes.

## Getting Started

### Prerequisites
- Python 3.7+
- Numpy
- Matplotlib
- Pymonntorch

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/NeuralPopulation.git
```
Navigate to the project directory and install dependencies:
```bash
cd NeuralPopulation
pip install -r requirements.txt
```

### Usage
0. **All together** : Explore all the different parts of the project at once and play around to find more about its features
   ```bash
   CNS-P02-Notebook.ipynb
   ```

2. **Synaptic Connectivity Schemes**: Explore different connectivity patterns using provided scripts.
   ```bash
   python synaptic_connectivity.py
   ```

3. **Balanced Network Design**: Design and analyze balanced networks.
   ```bash
   python balanced_network.py
   ```

4. **Decision-Making Dynamics**: Simulate decision-making processes.
   ```bash
   python decision_making.py
   ```

## Contributing and Stars
Contributions are welcome! Please fork and star the repository and submit a pull request with your enhancements or bug fixes. Ensure that your code adheres to the existing style guidelines and includes appropriate tests.

## Acknowledgments
Special thanks to Professor Mohammad Ganjtabesh for providing this project for us in 'computational neuroscience' course in faculty of Computer science at University of Tehran.

---

This repository aims to be a valuable resource for researchers and enthusiasts in computational neuroscience, providing tools and models to understand the complex behavior of neural networks. Feel free to explore, contribute, and expand the horizons of neural population dynamics!
