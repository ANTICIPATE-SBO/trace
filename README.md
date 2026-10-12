

# TRACE: Trajectory-based Clustering for Explaining Multi-Objective Reinforcement Learning
Developed at Vrije Universiteit Brussel (VUB) as part of the FWO-SBO project [ANTICIPATE](https://anticipate.vub.be/).

**Paper:** [Multi-Objective Decision Making Workshop @ IJCAI-ECAI 2026](https://openreview.net/pdf?id=pCWrKbpZP5) <br>
**Dependencies:** Python 3.11<br>
**Contact:** Emily Palaska (emily.palaska@vub.be)<br>
**License:** GNU General Public License v3.0

<p align="center">
  <img src="plots/logos/vub_ai_lab.png" height="60px" />
  <img src="plots/logos/fwo_kleur.png" height="60px" />
  <img src="plots/logos/ua_logo.jpg" height="60px" />
  <img src="plots/logos/logo-uzbrussel.jpg" height="60px" />
</p>


## ⚡ Quickstart
Clone this repository with `git clone https://github.com/ANTICIPATE-SBO/trace`<br>
Install requirements with `pip install -r requirements.txt`<br>


## ⚙️ Example of usage

```python
import numpy as np
from trace.core import TrajectoryManager
from trace.behavior import EmpiricalDistribution, distance_matrix, apply_quantization, quantization_mapping
from trace.clustering import k_medoids
from trace.visuals import temporal_alignment, trajectory_grid

# Trajectory loading from JSON file (see 'data/' directory)
manager = TrajectoryManager('dst-conc').load('ipro', pareto=True)

# Behavior features
obs, acs, rew = manager.conditioning_features()
if 'quant_obs' in manager.metadata:
    bins = manager.metadata['quant_bins']
    obs = apply_quantization(obs, quantization_mapping(obs, method='eq', bins=bins))
models = [EmpiricalDistribution(manager.metadata).fit(o, a) for o, a in zip(obs, acs)]
features = distance_matrix(models, metric='kl')

# Clustering algorithm
k = 5
labels, medoids = k_medoids(features, k=k)
clusters = [manager.subset(np.array(labels) == l) for l in range(k)]
medoid_manager = manager.subset([i in medoids for i in range(len(manager))])

# Plotting results
for c, cluster in enumerate(clusters):
    temporal_alignment(cluster, title=f'Cluster {c}').show()
    trajectory_grid(cluster, title=f'Cluster {c}').show()
trajectory_grid(medoid_manager, title='Medoids').show()
```

## 🧠 About
TRACE analyzes pareto-optimal trajectories collected by SOTA MORl policies and ground truth algorithms with the goal of
clustering them based in their behavior. The produced groups of trajectories are presented through interpretable
explanations and graphs. In this way, TRACE manages to summarize large amounts of trajectory data into intuitive
strategy insights for decision makers. It consists of five modules:
- **behavior**: empirical conditioning, behavior modeling, distance features and metrics
- **clustering**: k-means, k-medoids, gaussian/dirichlet mixture models, spectral
- **core**: data flow management, mathematical methods
- **policies**: ground truth algorithms and modified environments. 
- **visuals**: plotting functions mainly to identify patterns from the formed clusters

An abstract high-level flow chart of the mechanism:
<p align="center">
    <img src="plots/sketches/pipeline.png" width="500"/> <br>
    <img src="plots/sketches/trajectory.png" width="500"/>
</p>

## 💭 Behavior Modeling

To derive valuable behavior features for clustering, we first quantize the state space to have equal values per bin.
Then, to incorporate state-dependent structure into the
behavioral representation, we condition actions on the states in which they are performed. Finally, we compare two 
trajectories using the kl divergence on their common (quantized) state's empirical distributions.

<p align="center"> <img src="plots/sketches/conditioned.png" width="500" /> </p>

