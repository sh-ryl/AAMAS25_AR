import enum


class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    COLLECT = 4
    CRAFT = 5
    NO_OP = 6


class Material(enum.Enum):
    WOOD = "wood"
    IRON = "iron"
    GRASS = "grass"
    GEM = "gem"
    GOLD = "gold"


class Structure(enum.Enum):
    WORKBENCH = "workbench"
    TOOLSHED = "toolshed"
    FACTORY = "factory"


class Item(enum.Enum):
    AXE = "axe"
    BED = "bed"
    BRIDGE = "bridge"
    CLOTH = "cloth"
    PLANK = "plank"
    ROPE = "rope"
    STICK = "stick"


SPRITES = {
    Material.WOOD: '|',
    Material.IRON: '=',
    Material.GRASS: '#',
    Material.GEM: '@',
    Material.GOLD: '*',
    Structure.WORKBENCH: 'W',
    Structure.TOOLSHED: 'T',
    Structure.FACTORY: 'F'
}

REWARDABLE_ITEMS = {item.value for item in Item}

RECIPES = {
    Item.AXE: (Structure.TOOLSHED, {Material.IRON: 1, Item.STICK: 1}),
    Item.BED: (Structure.WORKBENCH, {Material.GRASS: 1, Item.PLANK: 1}),
    Item.BRIDGE: (Structure.FACTORY, {Material.IRON: 1, Material.WOOD: 1}),
    Item.CLOTH: (Structure.FACTORY, {Material.GRASS: 1}),
    Item.PLANK: (Structure.TOOLSHED, {Material.WOOD: 1}),
    Item.ROPE: (Structure.TOOLSHED, {Material.GRASS: 1}),
    Item.STICK: (Structure.WORKBENCH, {Material.WOOD: 1}),
}

RAW_RECIPES = {
    Item.AXE: {Material.IRON: 1, Material.WOOD: 1},
    Item.BED: {Material.GRASS: 1, Material.WOOD: 1},
    Item.BRIDGE: {Material.IRON: 1, Material.WOOD: 1},
    Item.CLOTH: {Material.GRASS: 1},
    Item.PLANK: {Material.WOOD: 1},
    Item.ROPE: {Material.GRASS: 1},
    Item.STICK: {Material.WOOD: 1},
}

# # graphic representation
# _sprites = {
#     "wood": '|',
#     "iron": '=',
#     "grass": '#',
#     "gem": '@',
#     "gold": '*',
#     "workbench": 'W',
#     "toolshed": 'T',
#     "factory": 'F'
# }

# _rewardable_items = [
#     "axe",
#     "bed",
#     "bridge",
#     "cloth",
#     "gem",
#     "gold",
#     "grass",
#     "iron",
#     "plank",
#     "rope",
#     "stick",
#     "wood"
# ]

# _recipes = {
#     "axe": ("toolshed", {"iron": 1, "stick": 1}),
#     "bed": ("workbench", {"grass": 1, "plank": 1}),
#     "bridge": ("factory", {"iron": 1, "wood": 1}),
#     "cloth": ("factory", {"grass": 1}),
#     "plank": ("toolshed", {"wood": 1}),
#     "rope": ("toolshed", {"grass": 1}),
#     "stick": ("workbench", {"wood": 1})
# }

# _raw_recipes = {
#     "axe": {"iron": 1, "wood": 1},
#     "bed": {"grass": 1, "wood": 1},
#     "bridge": {"iron": 1, "wood": 1},
#     "cloth": {"grass": 1},
#     "plank": {"wood": 1},
#     "rope": {"grass": 1},
#     "stick": {"wood": 1}
# }
