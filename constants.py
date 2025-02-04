import enum


class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    COLLECT = 4
    CRAFT = 5
    NO_OP = 6


SPRITES = {
    "wood": '|',
    "iron": '=',
    "grass": '#',
    "gem": '@',
    "gold": '*',
    "workbench": 'W',
    "toolshed": 'T',
    "factory": 'F'
}

REWARDABLE_ITEMS = [
    "axe",
    "bed",
    "bridge",
    "cloth",
    "gem",
    "gold",
    "grass",
    "iron",
    "plank",
    "rope",
    "stick",
    "wood"
]

RECIPES = {
    "axe": ("toolshed", {"iron": 1, "stick": 1}),
    "bed": ("workbench", {"grass": 1, "plank": 1}),
    "bridge": ("factory", {"iron": 1, "wood": 1}),
    "cloth": ("factory", {"grass": 1}),
    "plank": ("toolshed", {"wood": 1}),
    "rope": ("toolshed", {"grass": 1}),
    "stick": ("workbench", {"wood": 1})
}

RAW_RECIPES = {
    "axe": {"iron": 1, "wood": 1},
    "bed": {"grass": 1, "wood": 1},
    "bridge": {"iron": 1, "wood": 1},
    "cloth": {"grass": 1},
    "plank": {"wood": 1},
    "rope": {"grass": 1},
    "stick": {"wood": 1}
}

UNLIMITED_INV = {"wood": 999,
                 "iron": 999,
                 "grass": 999,
                 "gem": 999,
                 "gold": 999
                 }

LIMITED_INV = {"wood": 1,
               "iron": 1,
               "grass": 1,
               "gem": 999,
               "gold": 999
               }
