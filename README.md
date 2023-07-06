# Allsight_Sim : An Open-source Simulator for Allsight Tactile Sensor

## Overview
This package provides a simulator for a compact 3D strcutured vision-based Allsight finger sensor. It provides the Allsight model integration with [TACTO](https://github.com/facebookresearch/tacto) API simulation and a data collection process for robotic in-hand manipulation use cases.
For more information refer to the corresponding paper **edit** ...

---
## Installation

The code has been tested on:
- Ubuntu 18 / 20 
- python >= 3.6

Clone the repository:

```bash
git clone git@github.com:osheraz/allsight_sim.git
cd allsight_sim
```

Install the dependencies:

```bash
pip install -r requirements/requirements.txt
```

---

## Usage 

- [experiments/00_demo_pybullet_allsight.py](experiments/00_demo_pybullet_allsight.py): rendering RGB and Depth readings with Allsight sensor.

**gif**

NOTE: make sure ```summary``` param at the defaults list on ```experiment.yaml``` is set to ```demo```, and adjust the params on the ```summary/demo.yaml``` as you like.

- [experiments/01_collect_data_sim.py](experiments/01_collect_data_sim.py): rendering RGB and Depth readings with Allsight sensor.

**gif**

NOTE: make sure ```summary``` param at the defaults list on ```experiment.yaml``` is set to ```collect_data```, and adjust the params on the ```summary/collect_data.yaml``` as you like.







