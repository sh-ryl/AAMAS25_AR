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
# {"cloth": 0.1, "stick": 0.9}
# {"cloth": 0.3, "stick": 0.7}
# {"cloth": 0.5, "stick": 0.5}
# {"cloth": 0.7, "stick": 0.3}
# {"cloth": 0.9, "stick": 0.1}

# Belief
# {"iron": 0.7, "wood": -1, "grass": 1}
# {"iron": 0.7, "wood": 1, "grass": -1}

# Skill: {"axe": 1, "bridge": 0.8}

# Modified Q-Function: {"cloth": 0, "stick": 0}
# endregion

scenarios["train"] = {}

scenarios["train"]["attr_sets"] = [
    {"cloth": 0.9, "stick": 0.1}
]

scenarios["eval"] = {}
scenarios["eval"]["attr_sets"] = [
    {"cloth": 0.9, "stick": 0.1}
]
