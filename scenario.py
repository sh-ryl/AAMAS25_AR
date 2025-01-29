import constants


scenarios = {}


# Defaults
scenarios["default"] = {}

# by default ingredients won't regenerate once its collected
scenarios["default"]["regeneration"] = False

scenarios["default"]["starting_items"] = [{}, {}]

scenarios["default"]["num_spawned"] = {
    "wood": 4,
    "iron": 4,
    "grass": 4,
    "gem": 2,
    "gold": 2,
    "workbench": 2,
    "toolshed": 2,
    "factory": 2,
}

scenarios["default"]["hidden_items"] = [
    ["wood", "grass"]
]

# RL training/testing scenario

# region WEIGHT
# Goal preferences
# {"gem": 0.1, "gold": 0.9}
# {"gem": 0.3, "gold": 0.7}
# {"gem": 0.5, "gold": 0.5}
# {"gem": 0.7, "gold": 0.3}
# {"gem": 0.9, "gold": 0.1}

# Belief
# {"iron": 0.7, "wood": -1, "grass": 1}
# {"iron": 0.7, "wood": 1, "grass": -1}

# Skill: {"axe": 1, "bridge": 0.8}

# Modified Q-Function: {"cloth": 0, "stick": 0}
# endregion

scenarios["train"] = {}

scenarios["train"]["goal_sets"] = [
    {"gem": 0, "gold": 0}
]

scenarios["eval"] = {}
scenarios["eval"]["goal_sets"] = [
    {"axe": 1, "bridge": 0.7}
]
