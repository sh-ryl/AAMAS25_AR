# Beyond Goal Recognition: A Reinforcement Learning-based Approach to Inferring Agent Behaviour

Source code for the AAMAS-25 paper *Beyond Goal Recognition: A Reinforcement Learning-based Approach to Inferring Agent Behaviour*,
by Sheryl Mantik, Michael Dann, Minyi Li, Huong Ha, and Julie Porteous.

Code is adapted from *Multi-Agent Intention Recognition and Progression*, by Michael Dann, Yuan Yao, Natasha Alechina, Brian Logan, Felipe Meneguzzi and John Thangarajah.

Feel free to email Sheryl at sheryl.mantik@student.rmit.edu.au if you have any issues getting the code to run.

## Installing the Requirements

Code is tested on Python 3.9.4.
Required packages can be found under 'requirements.txt'.

## Model Training

To train a new RL policy, run:

```python python_agent.py train```

The goal items for the RL policy can be configured in scenario.py (line 32).

Agent scores are automatically logged to mod/*flags_during_training*/training_scores.csv.
Flags during training are the ones specified when running the python script.

## Execution

For each attribute, we have different setup tied to it. More descriptions can be found in our paper Section 4.2 (pg. 1432-1433)

We set which type of agent we want to observe by adjusting scenario.py, and adding the items listed in scenarios[”eval”][”attr_sets”]. We provide the weights we used in our experiments in each attribute. An example how the content of the list should look like:

```python
scenarios["eval"]["attr_sets"] = [
    {"cloth": 0.9, "stick": 0.1}
]
```

### Preferences
To run the experiment,
1. Choose 1 weight to be used in scenario.py before running the experiment:
```python
{"cloth": 0.1, "stick": 0.9}
{"cloth": 0.3, "stick": 0.7}
{"cloth": 0.5, "stick": 0.5}
{"cloth": 0.7, "stick": 0.3}
{"cloth": 0.9, "stick": 0.1}
```

2. Run this command through terminal:
```python python_agent.py eval AR```

### Beliefs
To run the experiment,
1. Choose 1 weight to be used in scenario.py before running the experiment:
```python
{"iron": 0.7, "wood": -1, "grass": 1}
{"iron": 0.7, "wood": 1, "grass": -1}
```

2. Run this command through terminal:
```python python_agent.py eval AR belief```

### Ability Level
To run the experiment,
1. Set this weight to be used in scenario.py before running the experiment:
```python
{"axe": 1, "bridge": 0.8}
```

2. Run this command through terminal:
```python python_agent.py eval AR ability incentive```
