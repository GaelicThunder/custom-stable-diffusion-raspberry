import random

def generate_prompt():
    if random.randint(0,1) == 0:
        return str(random.choice(actor)+","+random.choice(style)+","+random.choice(style)+","+random.choice(job))
    return str(random.choice(actor)+","+random.choice(style)+","+random.choice(style)+","+random.choice(job)+","+random.choice(special_powers))

actor = [
    "girl",
    "boy",
    "cat",
    "dog",
    "demon",
    "angel",
    "ghost",
    "robot",
    "alien",
    "vampire",
    "wizard",
    "warrior",
    "knight",
    "princess",
    "prince",
    "king",
    "queen",
    "elder",
    "witch",
    "ninja",
    "samurai",
    "detective",
    "scientist",
    "student",
    "teacher",
    "chef",
    "pilot",
    "mechanic",
    "musician",
    "artist",
    "writer",
    "librarian",
    "gardener",
    "farmer",
    "sailor",
    "pirate",
    "mermaid",
    "dragon",
    "phoenix",
    "unicorn",
    "goblin",
    "elf",
    "dwarf",
    "giant",
    "fairy",
    "ogre",
    "zombie",
    "mummy",
    "werewolf",
    "cyborg",
    "ghost",
    "superhero",
    "villain",
    "monster",
    "goddess",
    "god",
    "baker",
    "butler",
    "maid",
    "doctor",
    "nurse",
    "clown",
    "magician",
    "acrobat",
    "jester",
    "puppet",
    "doll",
    "shadow",
    "spirit",
    "idol",
    "racer",
    "gambler",
    "thief",
    "assassin",
    "shapeshifter",
    "time traveler",
    "alien",
    "astronaut",
    "android",
    "clone",
    "dinosaur",
    "pharaoh",
    "empress",
    "emperor",
    "jungle boy",
    "jungle girl",
    "panda",
    "bear",
    "eagle",
    "hawk",
    "raven",
    "wolf",
    "fox",
    "lion",
    "tiger",
    "cheetah",
    "panther",
    "jaguar",
    "leopard",
    "mouse",
    "rat",
    "rabbit",
    "deer",
    "owl",
    "snake",
    "fish",
    "shark",
    "octopus",
    "squid",
    "crab",
    "lobster",
    "starfish",
    "seahorse",
    "dolphin",
    "whale"
]

style = [
    "90's style",
    "80's style",
    "70's style",
    "60's style",
    "50's style",
    "cyberpunk",
    "steampunk",
    "fantasy",
    "sci-fi",
    "gothic",
    "victorian",
    "medieval",
    "ancient",
    "futuristic",
    "post-apocalyptic",
    "prehistoric",
    "traditional",
    "modern",
    "abstract",
    "minimalist",
    "baroque",
    "realistic",
    "surreal",
    "chibi",
    "magical girl",
    "mecha",
    "space opera",
    "moe",
    "beautiful",
    "weird",
    "standard",
    "ecchi",
    "cartoon",
    "CGI",
    "shoujo",
    "shounen",
    "seinen",
    "josei",
    "kodomo",
    "harem",
    "slice of life",
    "military",
    "sports",
    "school",
    "supernatural",
    "horror",
    "mystery",
    "psychological",
    "romantic",
    "action",
    "adventure",
    "comedy",
    "drama",
    "tragedy",
    "magic",
    "superpower",
    "martial arts",
    "music",
    "game",
    "historical",
    "parody",
    "samurai",
    "police",
    "mecha",
    "vampire",
    "yaoi",
    "yuri",
    "hentai",
    "demons",
    "dystopian",
    "post-apocalyptic",
    "alternate reality",
    "space",
    "reverse harem",
    "isekai",
    "mahou shoujo",
    "otome",
    "gore",
    "noir",
    "espionage",
    "superhero",
    "mythology",
    "fairy tale",
    "paranormal",
    "occult",
    "neo-noir",
    "cyber noir",
    "manga",
    "light novel",
    "visual novel",
    "doujinshi",
    "manhwa",
    "manhua",
    "webtoon",
    "4-koma",
    "shoujo ai",
    "shounen ai",
    "mecha musume",
    "kemonomimi",
    "yokai",
    "bishoujo",
    "bishounen",
    "chibi",
    "deformed",
    "super deformed",
    "gender bender",
    "hikikomori",
    "nekomimi",
    "usagi",
    "kitsune",
    "tanuki",
    "oni",
    "tengu",
    "kappa",
    "amazing world",
    "cute girls doing cute things",
    "ugly bastard",
    "netorare",
    "guro",
    "netori",
    "vanilla"
]

job = [
    "warrior",
    "ice maker",
    "driver",
    "chef",
    "baker",
    "pilot",
    "mechanic",
    "musician",
    "artist",
    "writer",
    "librarian",
    "gardener",
    "farmer",
    "sailor",
    "pirate",
    "detective",
    "scientist",
    "student",
    "teacher",
    "doctor",
    "nurse",
    "clown",
    "magician",
    "acrobat",
    "jester",
    "butler",
    "maid",
    "policeman",
    "fireman",
    "soldier",
    "knight",
    "samurai",
    "ninja",
    "assassin",
    "thief",
    "blacksmith",
    "carpenter",
    "miner",
    "fisherman",
    "hunter",
    "trapper",
    "explorer",
    "adventurer",
    "hermit",
    "monk",
    "priest",
    "shaman",
    "witch doctor",
    "fortune teller",
    "archer",
    "alchemist",
    "astronomer",
    "bard",
    "beastmaster",
    "berserker",
    "blackguard",
    "cavalier",
    "chrono mage",
    "cleric",
    "conjuror",
    "crusader",
    "cyber knight",
    "druid",
    "elementalist",
    "enchanter",
    "engineer",
    "geomancer",
    "gunslinger",
    "illusionist",
    "invoker",
    "juggernaut",
    "lancer",
    "marauder",
    "necromancer",
    "paladin",
    "ranger",
    "rogue",
    "runemaster",
    "sage",
    "shadowmancer",
    "shapeshifter",
    "sorcerer",
    "spellblade",
    "summoner",
    "thunder lord",
    "time traveler",
    "tracker",
    "vampire hunter",
    "warden",
    "warlock",
    "witch",
    "wizard",
    "zookeeper"
]

place = [
    "forest",
    "lake",
    "mountain",
    "ocean",
    "desert",
    "plains",
    "tundra",
    "island",
    "canyon",
    "swamp",
    "jungle",
    "volcano",
    "river",
    "waterfall",
    "glacier",
    "cave",
    "beach",
    "valley",
    "hill",
    "cliff",
    "plateau",
    "dune",
    "creek",
    "pond",
    "spring",
    "meadow",
    "ridge",
    "bay",
    "gulf",
    "fjord",
    "lagoon",
    "marsh",
    "bog",
    "delta",
    "reef",
    "strait",
    "cavern",
    "grotto",
    "crater",
    "savanna",
    "steppe",
    "taiga",
    "tropics",
    "wilderness",
    "woodland",
    "rainforest",
    "bamboo forest",
    "cherry blossom grove",
    "pine forest",
    "snowfield",
    "glade",
    "garden",
    "orchard",
    "vineyard",
    "farm",
    "ranch",
    "pasture",
    "field",
    "prairie",
    "heath",
    "moor",
    "oasis",
    "castle",
    "village",
    "city",
    "town",
    "countryside",
    "metropolis",
    "slum",
    "suburb",
    "downtown",
    "seaside town",
    "port",
    "harbor",
    "ruin",
    "temple",
    "shrine",
    "church",
    "cathedral",
    "abbey",
    "monastery",
    "theatre",
    "museum",
    "library",
    "school",
    "university",
    "hospital",
    "prison",
    "cemetery",
    "graveyard",
    "crypt",
    "dungeon",
    "fortress",
    "watchtower",
    "lighthouse",
    "mill",
    "factory",
    "bakery",
    "brewery",
    "winery",
    "distillery",
    "smithy",
    "workshop",
    "barn",
    "stables",
    "inn",
    "tavern",
    "pub",
    "brothel",
    "casino",
    "marketplace",
    "bazaar",
    "emporium",
    "mall",
    "arcade",
    "amusement park",
    "circus",
    "zoo",
    "aquarium",
    "planetarium",
    "observatory",
    "stadium",
    "arena",
    "colosseum",
    "racecourse",
    "golf course",
    "park",
    "garden",
    "orchard",
    "vineyard",
    "greenhouse",
    "nursery",
    "flower shop",
    "bookstore",
    "library",
    "museum",
    "art gallery",
    "concert hall",
    "opera house",
    "recording studio",
    "film studio",
    "dance studio",
    "theater",
    "laboratory",
    "workshop",
    "factory",
    "power plant",
    "hospital",
    "pharmacy",
    "veterinary clinic",
    "fire station",
    "police station",
    "post office",
    "bank",
    "courthouse",
    "embassy",
    "city hall",
    "parliament",
    "capitol",
    "palace",
    "mansion",
    "villa",
    "cottage",
    "farmhouse",
    "bungalow",
    "apartment building",
    "skyscraper",
    "hotel",
    "motel",
    "bed and breakfast",
    "hostel",
    "campsite",
    "caravan park",
    "trailer park",
    "resort",
    "spa",
    "hot spring",
    "beach house",
    "mountain cabin",
    "hunting lodge",
    "castle",
    "fort",
    "outpost",
    "bunker",
    "bomb shelter",
    "spaceship",
    "space station",
    "moon base",
    "planet",
    "galaxy",
    "universe",
    "multiverse",
    "dimension",
    "realm",
    "kingdom",
    "empire",
    "federation",
    "republic",
    "dictatorship",
    "monarchy",
    "theocracy",
    "anarchy",
    "colony",
    "protectorate",
    "tribe",
    "clan",
    "family",
    "community",
    "society",
    "culture",
    "civilization",
    "species",
    "race",
    "nation",
    "country",
    "state",
    "province",
    "county",
    "city",
    "town",
    "village",
    "neighborhood",
    "district",
    "region",
    "zone",
    "area",
    "sector",
    "quarter",
    "block",
    "street",
    "avenue",
    "boulevard",
    "lane",
    "alley",
    "path",
    "trail",
    "road",
    "highway",
    "freeway",
    "motorway",
    "bridge",
    "tunnel",
    "pass",
    "crossroads",
    "junction",
    "roundabout",
    "traffic circle",
    "intersection",
    "corner",
    "turn",
    "detour",
    "route",
    "way",
    "course",
    "direction",
    "line",
    "orbit",
    "trajectory",
    "channel",
    "passage",
    "corridor",
    "hallway",
    "staircase",
    "escalator",
    "elevator",
    "ladder",
    "ramp",
    "slope",
    "slide",
    "stairway",
    "doorway",
    "entrance",
    "exit",
    "portal",
    "gate",
    "fence",
    "wall",
    "barrier",
    "hurdle",
    "obstacle",
    "challenge",
    "problem",
    "issue",
    "difficulty",
    "hardship",
    "struggle",
    "conflict",
    "battle",
    "war",
    "revolution",
    "rebellion",
    "riot",
    "protest",
    "strike",
    "boycott",
    "demonstration",
    "march",
    "parade",
    "procession",
    "pilgrimage",
    "journey",
    "trip",
    "tour",
    "visit",
    "vacation",
    "holiday",
    "break",
    "pause",
    "stop",
    "rest",
    "relaxation",
    "leisure",
    "pleasure",
    "enjoyment",
    "delight",
    "joy",
    "happiness",
    "bliss",
    "ecstasy",
    "paradise",
    "heaven",
    "utopia",
    "dream",
    "fantasy",
    "illusion",
    "hallucination",
    "delusion",
    "mirage",
    "vision",
    "spectacle",
    "display",
    "show",
    "performance",
    "concert",
    "festival",
    "feast",
    "banquet",
    "party",
    "reception",
    "wedding",
    "funeral",
    "memorial",
    "tribute",
    "homage",
    "praise",
    "compliment",
    "flattery",
    "applause",
    "cheer",
    "ovation",
    "encore",
    "celebration",
    "ceremony",
    "ritual",
    "tradition",
    "custom",
    "habit",
    "routine",
    "pattern",
    "cycle",
    "circle",
    "sphere",
    "globe",
    "world",
    "earth",
    "planet",
    "star",
    "sun",
    "moon",
    "galaxy",
    "universe",
    "infinity",
    "eternity",
    "forever",
    "timeless",
    "endless",
    "boundless",
    "limitless",
    "unlimited",
    "infinite",
    "eternal",
    "perpetual",
    "constant",
    "continuous",
    "unending",
    "everlasting",
    "immortal",
    "undying",
    "indestructible",
    "invincible",
    "unbeatable",
    "unconquerable",
    "undefeated",
    "victorious",
    "triumphant",
    "successful",
    "prosperous",
    "flourishing",
    "thriving",
    "growing",
    "developing",
    "expanding",
    "advancing",
    "progressing",
    "evolving",
    "changing",
    "transforming",
    "metamorphosing",
    "transmuting",
    "transcending",
    "ascending",
    "rising",
    "soaring",
    "flying",
    "gliding",
    "sailing",
    "drifting",  
]

special_powers = [
    "It's raining and her power is controlling the elements",
    "She's enveloped by an aura of electricity",
    "Her body transforms into a slimy substance",
    "She has the ability to manipulate shadows",
    "Her eyes glow with a supernatural light",
    "Her voice can hypnotize anyone who hears it",
    "She can manipulate the fabric of reality",
    "She can move objects with her mind",
    "Her skin is as hard as diamond",
    "She can heal any injury in seconds",
    "She can become invisible at will",
    "She can phase through solid objects",
    "She can communicate with animals",
    "She can alter her appearance at will",
    "She can read and control minds",
    "She can teleport to any location",
    "She can see the future",
    "She can control time",
    "She can manipulate gravity",
    "She can control plants and make them grow at will",
    "She can generate and control fire",
    "She can control and shape water",
    "She can generate and control ice",
    "She can generate and control wind",
    "She can create illusions",
    "She can control technology",
    "She can absorb energy and redirect it",
    "She can replicate any physical action she sees",
    "She can control light and create hard-light constructs",
    "She can control sound waves",
    "She can generate and manipulate seismic energy",
    "She can manipulate her size and mass",
    "She can generate and control plasma",
    "She can transform into any animal",
    "She can control the weather",
    "She can generate force fields",
    "She can manipulate magnetic fields",
    "She has superhuman strength",
    "She has superhuman speed",
    "She has superhuman agility",
    "She has superhuman reflexes",
    "She has superhuman endurance",
    "She has superhuman intelligence",
    "She can fly",
    "She can breathe underwater",
    "She can survive in space",
    "She can survive in extreme temperatures",
    "She can survive without food or water",
    "She can survive without oxygen",
    "She can survive without sleep",
    "She can survive without aging",
    "She can survive without dying"]

