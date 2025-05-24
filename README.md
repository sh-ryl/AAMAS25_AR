# Beyond Goal Recognition: A Reinforcement Learning-based Approach to Inferring Agent Behaviour

Source code for the AAMAS-25 paper *Beyond Goal Recognition: A Reinforcement Learning-based Approach to Inferring Agent Behaviour*,
by Sheryl Mantik, Michael Dann, Minyi Li, Huong Ha, and Julie Porteous.

Code is adapted from *Multi-Agent Intention Recognition and Progression*, by Michael Dann, Yuan Yao, Natasha Alechina, Brian Logan, Felipe Meneguzzi and John Thangarajah.

## Installing the Requirements

Feel free to email Sheryl at sheryl.mantik@student.rmit.edu.au if you have any issues getting the code to run.

## Running

To recreate the results from the paper, run:

### Preferences
```python python_agent.py eval AR```
### Beliefs
```python python_agent.py eval AR belief```
### Ability Level
```python python_agent.py eval AR ability incentive```

Agent scores are automatically logged to results/cooperative_craft_world_dqn/results_*scenario_name*.csv.

To train a new RL policy, run:

```python python_agent.py train```

The goal items for the RL policy can be configured in scenario.py (line 32).
