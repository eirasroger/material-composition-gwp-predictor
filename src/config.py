"""
Project configuration: paths, hyperparameters, constants, and seed initialisation.
"""

from pathlib import Path
from typing import Dict, Set

import numpy as np
import torch


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset.json"

# ── Embedding backend ─────────────────────────────────────────────────────────
# Options:
#   "google_news" : local Word2Vec binary (.bin)
#   "fasttext"    : local/downloaded .vec
#   "custom_vec"  : user-provided .vec
EMBEDDING_BACKEND = "fasttext"

GOOGLE_NEWS_PATH  = BASE_DIR / "GoogleNews-vectors-negative300" / "GoogleNews-vectors-negative300.bin"

FASTTEXT_URL      = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
FASTTEXT_ZIP_PATH = BASE_DIR / "wiki-news-300d-1M.vec.zip"
FASTTEXT_DIR      = BASE_DIR / "wiki-news-300d-1M"
FASTTEXT_VEC_PATH = FASTTEXT_DIR / "wiki-news-300d-1M.vec"

CUSTOM_VEC_PATH   = BASE_DIR / "custom_vectors.vec"

# ── Output paths ──────────────────────────────────────────────────────────────
FIGURES_DIR        = BASE_DIR / "figures"

MODEL_PATH         = BASE_DIR / "ghg_model.pt"
DIAGNOSTICS_PATH   = BASE_DIR / "diagnostics_ghg.json"
TRAINING_PLOT_PATH = FIGURES_DIR / "training_curves_ghg.png"
PRED_SCATTER_PATH  = FIGURES_DIR / "pred_vs_actual_ghg.png"
RESIDUALS_PATH     = FIGURES_DIR / "residuals_ghg.png"

# ── Model hyperparameters ─────────────────────────────────────────────────────
EMBED_DIM          = 300
HIDDEN_DIMS        = [256, 128]
DROPOUT            = 0.25
WEIGHT_DECAY       = 3e-4
LR                 = 1e-3
EPOCHS             = 200
BATCH_SIZE         = 64
PATIENCE           = 20
RANDOM_SEED        = 42
SEEDS              = [42, 123, 456, 789, 2024]

TARGET_FIELD       = "total_ghg"

GHG_MIN            = 0.0
GHG_MAX            = 10.0

MIN_CATEGORY_COUNT = 10

# 1 circularity-origin + 4 end-of-life percentages
N_CIRC_FEATURES    = 5

# ── Material token normalisation ──────────────────────────────────────────────
MATERIAL_VARIATIONS: Dict[str, str] = {
    "aluminium": "aluminum",
    "braces":    "suspenders",
    "colour":    "color",
    "fibre":     "fiber",
    "flavour":   "flavor",
    "gaol":      "jail",
    "kerb":      "curb",
    "litre":     "liter",
    "mould":     "mold",
    "petrol":    "gasoline",
    "sulphur":   "sulfur",
    "tyre":      "tire",
    "woollen":   "woolen",


     # Misspellings of known materials
    "acrylnitril":     "acrylonitrile",
    "anydrite":        "anhydrite",
    "calcit":          "calcite",
    "chammotte":       "chamotte",
    "chamotte":        "refractory",       # or "fireclay"
    "kaoline":         "kaolin",
    "kaolins":         "kaolin",
    "pozzolona":       "pozzolana",
    "polurethane":     "polyurethane",
    "polyamid":        "polyamide",
    "polypropylen":    "polypropylene",
    "poliol":          "polyol",
    "polymide":        "polyimide",
    "elastomere":      "elastomer",
    "duroplast":       "thermoset",
    "thermoplast":     "thermoplastic",
    "styrol":          "styrene",
    "butadien":        "butadiene",
    "aramide":         "aramid",
    "hardner":         "hardener",
    "currative":       "curative",
    "nonvolatiles":    "solids",
    "fluidifying":     "plasticizer",
    "fluidizer":       "plasticizer",
    "superfluidifying":"superplasticizer",
    "liquifiers":      "plasticizers",
    "redispersible":   "polymer",
    "feldespars":      "feldspar",
    "feldspath":       "feldspar",
    "magnezite":       "magnesite",
    "sulfoalminate":   "sulfoaluminate",
    "sulfoaluminate":  "calcium aluminate",
    "sulphonate":      "sulfonate",
    "sulphonic":       "sulfonic",
    "lignite":         "lignite",
    "ligno":           "lignin",
    "lignosulfonate":  "lignosulfonate",
    "lignosulphonate": "lignosulfonate",
    "sawnwood":        "lumber",
    "glassfiber":      "fiberglass",
    "stonewool":       "rockwool",
    "hotmelt":         "adhesive",
    "spunbond":        "nonwoven",
    "bicomponent":     "composite",
    "coextruded":      "extruded",
    "granulates":      "granules",
    "briquetted":      "briquette",
    "regranulated":    "recycled",
    "prepainted":      "coated",
    "electrogalvanized":"galvanized",
    "anodise":         "anodize",
    "epoxidised":      "epoxidized",
    "bituminized":     "bituminous",
    "vinylester":      "vinyl ester",
    "vinylic":         "vinyl",
    "methylmethacrylate": "methyl methacrylate",
    "cycloaliphatic":  "aliphatic",
    "cyanoguanidin":   "cyanoguanidine",
    "dihydroxide":     "hydroxide",
    "epoxypropane":    "epoxide",
    "ethenylbenzene":  "styrene",
    "octabenzone":     "benzophenone",
    "isopropylidenediphenol": "bisphenol",
    "tetrabromobisphenol": "brominated epoxy",
    "morpholinopropylamine": "amine",
    "triethylenetetramine": "amine",
    "triethoxysilane": "silane",
    "phenylenebis":    "antioxidant",
    "dimethylaminomethyl": "amine",
    "polycarbodiimide": "carbodiimide",
    "polycarboxylate": "carboxylate",
    "polyetheramine":  "amine",
    "polyarylether":   "polymer",
    "polyphenyl":      "polyphenylene",
    "polyphenylene":   "polymer",
    "polyphenylsulfone": "polysulfone",
    "polytersulfone":  "polysulfone",
    "polysulphide":    "polysulfide",
    "polyoxymethylen": "polyoxymethylene",
    "polyoxymetylen":  "polyoxymethylene",
    "polymethyl":      "polymethyl methacrylate",
    "polymethylene":   "polyethylene",
    "polyisocyanurate":"polyurethane",
    "polyacrylamid":   "polyacrylamide",
    "polyalcohol":     "polyvinyl alcohol",
    "polyetylentereftalat": "polyethylene terephthalate",
    "polypropene":     "polypropylene",
    "monopropylene":   "propylene glycol",
    "diisononyl":      "plasticizer",
    "dinch":           "plasticizer",
    "dinp":            "plasticizer",
    "dinonyl":         "plasticizer",
    "dotp":            "plasticizer",
    "diformate":       "formate",
    "diazole":         "imidazole",
    "isothiazolin":    "isothiazolinone",
    "omadine":         "isothiazolinone",
    "tbba":            "brominated epoxy",
    "compatibilizer":  "polymer",
    "coalescents":     "solvent",
    "entrainer":       "admixture",
    "nucleant":        "nucleating agent",
    "swellable":       "expansive",
    "waterbase":       "aqueous",
    "toughener":       "modifier",
    "spinfinish":      "lubricant",
    "backcoat":        "coating",
    "sealcoat":        "sealant",
    "sylancoat":       "silane",
    "groutcoat":       "grout",
    "fiberbinder":     "binder",   




       # French
    "alliage":         "alloy",
    "feldspath":       "feldspar",   # already above
    "lasur":           "glaze",
    "engobe":          "slip",       # ceramic slip/engobe

    # Italian
    "graniglia":       "granule",
    "ghiaietto":       "gravel",
    "pietrisco":       "aggregate",
    "sagomati":        "shaped",
    "induritore":      "hardener",
    "vergella":        "wire rod",





        "hdpe":   "polyethylene",
    "ldpe":   "polyethylene",
    "mdpe":   "polyethylene",
    "lldpe":  "polyethylene",
    "pehd":   "polyethylene",
    "pex":    "polyethylene",
    "xlpe":   "polyethylene",
    "ixpe":   "polyethylene",
    "pvac":   "polyvinyl acetate",
    "pva":    "polyvinyl alcohol",
    "pvb":    "polyvinyl butyral",
    "pmma":   "acrylic",
    "ptfe":   "polytetrafluoroethylene",
    "epdm":   "rubber",
    "fkm":    "fluoropolymer",
    "hnbr":   "rubber",
    "sbr":    "rubber",
    "tpu":    "polyurethane",
    "tpv":    "rubber",
    "pbt":    "polyester",
    "petg":   "polyester",
    "upvc":   "polyvinyl chloride",
    "ppma":   "acrylic",
    "ppo":    "polyphenylene",
    "mdi":    "isocyanate",
    "tdi":    "isocyanate",
    "muf":    "resin",
    "umf":    "resin",
    "hpl":    "laminate",
    "hdf":    "fiberboard",
    "gfrp":   "fiberglass",
    "cfrt":   "carbon fiber",
    "eaf":    "steel",
    "opc":    "cement",
    "ggbs":   "slag",
    "ggbfs":  "slag",
    "fgd":    "gypsum",
    "scms":   "slag",
    "leca":   "aggregate",
    "igu":    "glass",
    "mgo":    "magnesia",
    "naoh":   "sodium hydroxide",
    "zno":    "zinc oxide",
    "tiox":   "titanium dioxide",
    "nbo":    "niobium",
    "fesi":   "ferrosilicon",
    "fev":    "ferrovanadium",
    "nicr":   "nickel chromium",
    "cusi":   "copper silicon",
    "cazn":   "calcium zinc",
    "simn":   "silicon manganese",
    "znal":   "zinc aluminum",
    "znsn":   "zinc tin",
    "crca":   "steel",



    "calcium aluminate":           "aluminate",
"brominated epoxy":            "epoxy",
"methyl methacrylate":        "methacrylate",
"polyvinyl chloride":         "chloride",
"polyvinyl acetate":          "acetate",
"polyvinyl alcohol":          "alcohol",
"polyvinyl butyral":          "butyral",
"polyethylene terephthalate": "terephthalate",
"polymethyl methacrylate":    "methacrylate",
"nucleating agent":           "nucleating",
"wire rod":                   "rod",
"zinc oxide":                 "zinc",
"titanium dioxide":           "titanium",
"sodium hydroxide":           "sodium",
"carbon fiber":               "carbon",
"propylene glycol":           "propylene",
"vinyl ester":                "ester",
"nickel chromium":            "nickel",
"silicon manganese":          "silicon",
"copper silicon":             "copper",
"zinc aluminum":              "zinc",
"zinc tin":                   "tin",
"calcium zinc":               "zinc",
"ferrosilicon":               "silicon",
"ferrovanadium":              "vanadium",
"sulfoaluminate":             "aluminate",
"isothiazolinone":            "isothiazole",
"cyanoguanidine":             "guanidine",
"lignosulfonate":             "lignin",
"aluminoferrite":  "ferrite",
"aluzinc":         "zinc",
"amina":           "amine",
"aminopropyl":     "amine",
"anodize":         "anodized",
"chipboards":      "chipboard",
"cullets":         "cullet",
"formol":          "formaldehyde",
"hexogene":        "explosive",
"hydrophobizer":   "hydrophobic",
"molochite":       "mullite",
"octene":          "octane",
"paperliner":      "liner",
"pentrite":        "explosive",
"rocksand":        "sand",
"rozin":           "resin",
"shives":          "fiber",
"silikon":         "silicone",
"silikonlist":     "silicone",
"steelmake":       "steel",
"stretchwrap":     "film",
"styron":          "polystyrene",
"trass":           "pozzolana",
"trimeta":         "phosphate",
"sylvic":          "resin",
"microfiltered":   "filtered",
"biospersistent":  "persistent",
"socketing":       "fitting",
"alkans":          "alkane",
"aditives":        "additive",
"agron":           "argon",
"econyl":          "nylon",
"aquafil":         "nylon",
"accoya":          "pine",
"zamak":           "zinc",
"stellite":        "cobalt",
"aluzinc":         "aluminum",
"magnelis":        "zinc",
"corten":          "steel",
"armaflex":        "elastomer",
"foamglas":        "glass",
"tanalith":        "timber",
"santoprene":      "rubber",
"lumiflon":        "fluoropolymer",
"sikament":        "admixture",
"mapei":           "adhesive",
"mapelastic":      "polymer",
"penetron":        "cement",
"dynamon":         "admixture",
"kronospan":       "fiberboard",
"egger":           "fiberboard",
"basf":            "polymer",
"koppers":         "resin",
"sisecam":         "glass",
"illbruck":        "sealant",
"fibron":          "fiber",
"promatect":       "calcium",
"technoflame":     "intumescent",
"thermopine":      "timber",
"greentec":        "recycled",
"greencem":        "cement",
"ecocemento":      "cement",
"ecobase":         "polymer",
"ecoplanet":       "recycled",
"nomatec":         "polyurethane",
"lambdapor":       "polystyrene",
"evopreno":        "neoprene",
"insulborad":      "insulation",
"inertex":         "fiber",
"intertex":        "fiber",
"isomat":          "polymer",
"krystaline":      "cement",
"volcanite":       "gypsum",
"stonblend":       "aggregate",
"stonkote":        "coating",
"stonseal":        "sealant",
"supershield":     "coating",
"blueclad":        "cladding",
"converlight":     "aggregate",
"colorferox":      "pigment",
"marmocryl":       "acrylic",
"marvon":          "polymer",
"probase":         "resin",
"polidex":         "polymer",
"viscostar":       "admixture",
"masterease":      "admixture",
"mastersuna":      "admixture",
"getacore":        "acrylic",
"sansin":          "coating",
"wetfix":          "adhesive",
"xcarb":           "steel",
"biozinalium":     "zinc",
"styron":          "polystyrene",
"canastol":        "resin",
"sylvic":          "resin",
"rozin":           "resin",
"trimeta":         "phosphate",
"thiover":         "sulfur",
"dlk":             "lubricant",
"dsg":             "aggregate",
"fvp":             "fiber",
"mff":             "fiber",
"nwf":             "nonwoven",
"shg":             "glass",
"pns":             "naphthalene",
"petch":           "polyester",
"svhc":            "hazardous",
"ibc":             "container",
"mcpcb":           "aluminum",
"shives":          "fiber",
"bigbag":          "packaging",
"coex":            "extruded",
"baxab":           "polymer",
"cisi":            "silicate",
"beslow":          "admixture",
"aquron":          "admixture",
"cindol":          "lubricant",
"delvo":           "admixture",
"pfleider":        "aggregate",
"nygran":          "nylon",
"softpex":         "polyethylene",
"microduct":       "polyethylene",
"gluethread":      "adhesive",
"cartonfelt":      "cellulose",
"paperliner":      "paper",
"stretchwrap":     "film",
"biospersistent":  "fiber",
"hexogene":        "explosive",
"pentrite":        "explosive",
"alkans":          "alkane",
"amina":           "amine",
"butylver":        "butyl",
"formol":          "formaldehyde",
"molochite":       "mullite",
"cullets":         "glass",
"chipboards":      "chipboard",
"rocksand":        "sand",
"steelmake":       "steel",
"colemanite":      "boron",
"aluminoferrite":  "ferrite",
"aminopropyl":     "amine",
"cyanoguanidine":  "guanidine",
"isothiazolinone": "biocide",
"lignosulfonate":  "lignin",
"hydrophobizer":   "silicone",
"trass":           "pozzolana",
"sulfoaluminate":  "aluminate",
"aditives":        "additive",
"agron":           "argon",
"anodize":         "anodized",
"chipboards":      "chipboard",
"hexogene":        "explosive",
"microfiltered":   "filtered",
"octene":          "octane",
"silikon":         "silicone",
"silikonlist":     "silicone",
"socketing":       "fitting",
"steelmake":       "steel",
"alkans":          "alkane",

}

STOP_WORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "has", "have", "in", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "there", "they", "this",
    "to", "was", "were", "which", "with",

    "type", "grade", "class", "kind", "form", "series", "model", "product",
    "item", "sample", "specification", "spec", "material", "component",
    "part", "version", "variant", "lot", "batch", "range"
}
# Reproducibility: applied on import so any module that touches torch / numpy
# downstream sees the same seeds as the original single-file script did.
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
