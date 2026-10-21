"""
engine.py - NutriAI meal-planning engine
========================================
Pipeline: dedup -> clinical filtering -> allergy exclusion -> diet preference
-> ranking (neural-network scorer or FAISS nearest-neighbor retrieval)
-> realistic meal assembly (meal templates + serving sizes) -> nutrient analysis
-> feedback (saved/skipped meals, Thompson-sampling bandit).
"""
from __future__ import annotations
import os, math, random, re, time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "food_snapshot.csv")

MACROS = ["energy_kcal","protein_g","carb_g","fat_g","fiber_g"]
MICROS = ["iron_mg","calcium_mg","vitamin_b12_ug","vitamin_d_ug","zinc_mg"]
EXTRA  = ["sugar_g","sodium_mg","potassium_mg","vitamin_c_mg","saturated_fat_g",
          "cholesterol_mg","magnesium_mg"]
NUTRIENTS = MACROS + MICROS + EXTRA

# RDA tables (adult), tailored by sex. Values approximate NIH DRIs.
RDA = {
    "male":   {"iron_mg":8,  "calcium_mg":1000,"vitamin_b12_ug":2.4,"vitamin_d_ug":15,
               "zinc_mg":11, "fiber_g":38,"potassium_mg":3400,"protein_g":56,
               "magnesium_mg":400,"vitamin_c_mg":90},
    "female": {"iron_mg":18, "calcium_mg":1000,"vitamin_b12_ug":2.4,"vitamin_d_ug":15,
               "zinc_mg":8,  "fiber_g":25,"potassium_mg":2600,"protein_g":46,
               "magnesium_mg":310,"vitamin_c_mg":75},
}
def rda_for(age, sex):
    base = dict(RDA.get(sex, RDA["female"]))
    if age and age >= 51:                 # older adults
        base["calcium_mg"] = 1200
        if sex == "female": base["iron_mg"] = 8
    return base

GERD_TRIGGERS = ["garlic", "onion","orange", "lemon", "lime", "grapefruit", "citrus","tomato","fried","spicy","caffeine","chocolate"]
ALLERGEN_CHOICES = ["milk","egg","fish","shellfish","tree_nut","peanut","soy",
                    "wheat","gluten"]
MEALS = ["Breakfast","Lunch","Dinner"]


# ---------------- Thompson-sampling bandit (reinforcement-learning technique) -----
class ThompsonBandit:
    """Contextual multi-armed bandit over food groups (arms).

    Each arm = a food group. We keep a Beta(alpha, beta) success/failure
    distribution per arm. On thumbs-up we increment that group's alpha; on
    thumbs-down we increment beta. To pick which group a flexible meal slot
    should favor, we Thompson-sample each arm (draw from its Beta) and rank by
    the samples - this naturally balances EXPLORING under-tried groups against
    EXPLOITING groups the user has liked. Persisted across sessions in the app.
    """
    def __init__(self, arms):
        self.alpha = {a: 1.0 for a in arms}   # prior successes (+1 smoothing)
        self.beta  = {a: 1.0 for a in arms}   # prior failures
    def update(self, arm, reward):
        if arm not in self.alpha:
            self.alpha[arm] = 1.0; self.beta[arm] = 1.0
        if reward > 0: self.alpha[arm] += 1.0
        else:          self.beta[arm]  += 1.0
    def sample(self, arm, rng):
        a, b = self.alpha.get(arm, 1.0), self.beta.get(arm, 1.0)
        # Beta sample via two Gammas (uses numpy's RNG seeded by `rng`)
        return float(np.random.default_rng(rng.randint(0, 2**31)).beta(a, b))
    def rank_groups(self, groups, rng):
        return sorted(groups, key=lambda g: self.sample(g, rng), reverse=True)
    def means(self):
        return {a: round(self.alpha[a]/(self.alpha[a]+self.beta[a]), 3)
                for a in self.alpha}


@dataclass
class Persona:
    name: str
    conditions: list
    allergens: list             # allergen tags to exclude
    diet: str                   # vegan/vegetarian/pescatarian/nonveg
    kcal_target: int = 2000
    age: int = 35
    sex: str = "female"
    lactose_free: bool = False
    no_pork: bool = False
    gluten_free: bool = False
    require_fish_meals: int = 0       # min fish/seafood meals across week
    sodium_daily_cap: int = 0         # 0 = none
    gi_max: int = 0                   # 0 = none
    religious: str = ""              # optional UI field; kept for app compatibility
    meal_diets: dict = field(default_factory=dict)  # optional per-meal diet preferences
    micro_priority: list = field(default_factory=list)


@dataclass
class ExclusionLog:
    clinical: dict = field(default_factory=dict)
    allergy: dict = field(default_factory=dict)
    diet: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)   # foods from meals the user skipped
    def counts(self):
        return {"clinical":len(self.clinical),"allergy":len(self.allergy),
                "diet":len(self.diet),"skipped":len(self.skipped)}



# Synthetic brand prefixes used in the food database (build_food_db.py).
DISPLAY_BRANDS = [
    "healthy harvest", "nature's best", "fresh direct", "purefoods", "meadow", "coastal",
    "simply pure", "heritage", "greenfield", "vital", "farmhouse", "earthgrown",
    "wholesome", "daily choice", "prime", "garden select", "sunrise", "golden acre",
    "organic valley co", "truenorth",
]
# Same brands after punctuation is stripped (used by _canonical_food_name)
CANONICAL_BRANDS = [b.replace("'", " ") for b in DISPLAY_BRANDS] + ["natures best"]


def _canonical_food_name(name: str) -> str:
    """Collapse branded/form variants into the same base food.

    Examples:
    - "Coastal Oats (family pack)" -> "oats"
    - "Fresh Direct Pumpkin seeds" -> "pumpkin seeds"
    - "Kale, raw" -> "kale"

    This is used for no-repeat logic, because the app should not treat different
    brands of the same food as different meals.
    """
    text = str(name or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)          # remove parenthetical forms
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s,%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove synthetic brand prefixes created in build_food_db.py.
    for b in CANONICAL_BRANDS:
        if text.startswith(b + " "):
            text = text[len(b):].strip()
            break

    # Drop preparation/packaging descriptors so repeated base foods are counted.
    descriptors = {
        "raw", "cooked", "baked", "canned", "steamed", "roasted", "prepared",
        "frozen", "dry", "plain", "whole", "firm", "unsweetened", "low", "sodium",
        "reduced", "fat", "value", "size", "family", "pack", "gluten", "free"
    }
    words = []
    for w in re.split(r"[\s,%-]+", text):
        if w and w not in descriptors:
            words.append(w)
    text = " ".join(words).strip()

    # Hand-normalize common variants.
    replacements = {
        "oat": "oats",
        "pumpkin seed": "pumpkin seeds",
        "chia seed": "chia seeds",
        "soy milk": "soy milk",
        "oat milk": "oat milk",
        "almond milk": "almond milk",
        "lactose milk": "lactose-free milk",
        "white rice": "white rice",
        "brown rice": "brown rice",
        "pasta": "pasta",
        "wheat pasta": "pasta",
        "pasta wheat": "pasta",
        "beef lean": "beef",
        "tuna in water": "tuna",
        "milk 2": "milk",
        "bell pepper red": "bell pepper",
    }
    return replacements.get(text, text or str(name).lower().strip())

# ---------------- realistic meals: roles and serving sizes ----------------
# The food database lists nutrients per 100 g. These tables say what part of a
# meal each food can play and how much of it a person would realistically eat.

# Seasonings are never a meal component on their own.
SEASONINGS = {"garlic", "onion", "lemon", "olive oil"}

# Serving range in grams for one item in one meal (US-style serving sizes).
SERVINGS = {
    # dairy and milks
    "milk": (240, 240), "milk 2": (240, 240), "lactose-free milk": (240, 240),
    "soy milk": (240, 240), "almond milk": (240, 240), "oat milk": (240, 240),
    "greek yogurt": (150, 250), "cottage cheese": (110, 225), "cheddar cheese": (20, 45),
    # breakfast grains and breads
    "oats": (40, 80), "wheat bread": (30, 90), "corn tortilla": (25, 100),
    # cooked grains and starches
    "brown rice": (120, 250), "white rice": (120, 250), "quinoa": (120, 220),
    "barley": (120, 220), "buckwheat": (150, 250), "pasta": (140, 250),
    "pasta wheat": (140, 250), "sweet potato": (130, 250),
    # proteins (cooked weight)
    "chicken breast": (100, 180), "turkey breast": (100, 170), "beef": (85, 170),
    "beef lean": (85, 170), "pork chop": (100, 170), "salmon": (100, 170),
    "cod": (110, 200), "tuna": (85, 140), "tuna in water": (85, 140),
    "sardines": (100, 100), "shrimp": (100, 170), "tofu": (100, 200),
    "tempeh": (85, 150), "lentils": (120, 220), "black beans": (120, 200),
    "chickpeas": (120, 200), "edamame": (100, 160), "egg": (50, 150),
    # nuts, seeds, spreads
    "almonds": (14, 42), "walnuts": (14, 28), "pumpkin seeds": (14, 28),
    "peanut butter": (16, 32), "chia seeds": (12, 24), "flaxseed": (10, 20),
    # fruit
    "apple": (90, 180), "banana": (60, 120), "orange": (70, 140), "kiwi": (75, 150),
    "blueberries": (75, 150), "strawberries": (100, 200), "grapes": (80, 150),
    "avocado": (50, 100),
    # vegetables
    "broccoli": (80, 180), "cauliflower": (80, 180), "green beans": (80, 160),
    "carrots": (60, 130), "bell pepper": (75, 150), "zucchini": (80, 180),
    "eggplant": (80, 180), "tomato": (80, 160), "cucumber": (80, 150),
    "kale": (50, 100), "spinach": (60, 120),
}

ROLE_FOODS = {
    "bfast_main": {"oats", "wheat bread", "greek yogurt", "cottage cheese", "egg", "buckwheat"},
    "fruit": {"apple", "banana", "orange", "kiwi", "blueberries", "strawberries", "grapes"},
    "protein": {"chicken breast", "turkey breast", "beef", "beef lean", "pork chop", "salmon",
                "cod", "tuna", "tuna in water", "sardines", "shrimp", "tofu", "tempeh",
                "lentils", "black beans", "chickpeas", "edamame", "egg"},
    "starch": {"brown rice", "white rice", "quinoa", "barley", "buckwheat", "pasta",
               "pasta wheat", "wheat bread", "corn tortilla", "sweet potato"},
    "veg": {"broccoli", "cauliflower", "green beans", "carrots", "bell pepper", "zucchini",
            "eggplant", "tomato", "cucumber", "kale", "spinach"},
    "snack": {"greek yogurt", "almonds", "walnuts", "pumpkin seeds", "apple", "banana",
              "orange", "grapes", "edamame", "cottage cheese", "peanut butter"},
}

# What goes with each breakfast main (so breakfasts look like real breakfasts).
_MILKS = ["milk", "milk 2", "lactose-free milk", "soy milk", "almond milk", "oat milk"]
_NUTS_SEEDS = ["almonds", "walnuts", "pumpkin seeds", "chia seeds", "flaxseed"]
BFAST_SIDES = {
    "oats": _MILKS + _NUTS_SEEDS + ["peanut butter"],
    "buckwheat": _MILKS + _NUTS_SEEDS,
    "greek yogurt": _NUTS_SEEDS,
    "cottage cheese": _NUTS_SEEDS,
    "wheat bread": ["peanut butter", "egg", "avocado", "cheddar cheese", "cottage cheese"],
    "egg": ["wheat bread", "avocado", "corn tortilla", "cheddar cheese"],
}
DEFAULT_BFAST_SIDES = _MILKS + _NUTS_SEEDS + ["avocado", "peanut butter"]

MEAL_SHARE = {"Breakfast": 0.27, "Lunch": 0.35, "Dinner": 0.38}
MEAL_TEMPLATES = {
    "Breakfast": [("bfast_main", 0.55), ("fruit", 0.20), ("bfast_side", 0.25)],
    "Lunch":     [("protein", 0.45), ("starch", 0.35), ("veg", 0.20)],
    "Dinner":    [("protein", 0.45), ("starch", 0.30), ("veg", 0.25)],
}
# Most times a base food should appear in one week, by role.
REPEAT_CAP = {"protein": 2, "starch": 3, "veg": 2, "fruit": 2, "bfast_main": 3,
              "bfast_side": 3, "snack": 2}

# Fallbacks for foods added to the database later.
_GROUP_DEFAULTS = {"Protein": ("protein", (85, 170)), "Grains": ("starch", (120, 220)),
                   "Vegetables": ("veg", (80, 160)), "Fruits": ("fruit", (80, 150)),
                   "Dairy": ("snack", (150, 250)), "Fats": (None, (5, 15))}


def serving_range(base, group, kcal100):
    if base in SERVINGS:
        return SERVINGS[base]
    if kcal100 > 450:                       # nut/seed-like: small handful
        return (14, 35)
    return _GROUP_DEFAULTS.get(str(group), (None, (40, 200)))[1]


def roles_for(base, group, kcal100):
    if base in SEASONINGS:
        return set()
    roles = {role for role, members in ROLE_FOODS.items() if base in members}
    if roles or base in SERVINGS:
        return roles
    role = _GROUP_DEFAULTS.get(str(group), (None, None))[0]
    if role is None:
        return set()
    return {"snack"} if kcal100 > 450 else {role}


# Everyday units: base -> (grams per unit, singular, plural, step)
HOUSEHOLD = {
    "egg": (50, "egg", "eggs", 1), "wheat bread": (30, "slice", "slices", 1),
    "corn tortilla": (25, "tortilla", "tortillas", 1), "sardines": (100, "can", "cans", 1),
    "banana": (120, "banana", "bananas", 0.5), "apple": (180, "apple", "apples", 0.5),
    "orange": (140, "orange", "oranges", 0.5), "kiwi": (75, "kiwi", "kiwis", 1),
    "milk": (240, "cup", "cups", 1), "milk 2": (240, "cup", "cups", 1),
    "lactose-free milk": (240, "cup", "cups", 1), "soy milk": (240, "cup", "cups", 1),
    "almond milk": (240, "cup", "cups", 1), "oat milk": (240, "cup", "cups", 1),
    "peanut butter": (16, "tbsp", "tbsp", 1), "chia seeds": (12, "tbsp", "tbsp", 1),
    "flaxseed": (10, "tbsp", "tbsp", 1), "almonds": (28, "oz", "oz", 0.5),
    "walnuts": (28, "oz", "oz", 0.5), "pumpkin seeds": (28, "oz", "oz", 0.5),
}


def snap_portion(base, grams, lo, hi):
    """Round a portion to a sensible amount (whole eggs, slices, half fruits, 5 g steps)."""
    if base in HOUSEHOLD:
        unit, _, _, step = HOUSEHOLD[base]
        n = max(step, round(grams / unit / step) * step)
        g = n * unit
        # keep within the serving range when a whole unit allows it
        while g > hi and n - step >= step:
            n -= step; g = n * unit
        while g < lo and g + step * unit <= hi:
            n += step; g = n * unit
        return float(g)
    return float(min(max(round(grams / 5.0) * 5, lo), hi))


def _fmt_amount(n):
    whole, frac = int(n), n - int(n)
    if abs(frac - 0.5) < 1e-6:
        return "½" if whole == 0 else f"{whole}½"
    return str(int(round(n)))


def portion_label(food):
    """'2 slices', '1 cup', '150 g' for a food dict in a plan."""
    base = food.get("_base_food") or _canonical_food_name(food.get("name", ""))
    g = float(food.get("portion_g", 0))
    if base in HOUSEHOLD and g > 0:
        unit, single, plural, _ = HOUSEHOLD[base]
        n = g / unit
        if abs(n * 2 - round(n * 2)) < 0.05:          # clean half-unit amounts only
            n = round(n * 2) / 2
            word = single if n <= 1 else plural
            return f"{_fmt_amount(n)} {word}"
    return f"{g:.0f} g"


def display_food_name(name):
    """Readable food name without synthetic brand prefixes or package notes.

    'Garden Select Quinoa (raw)' -> 'Quinoa'; 'Chicken breast, cooked' stays as is.
    """
    text = re.sub(r"\([^)]*\)", " ", str(name or ""))
    text = re.sub(r"\s+", " ", text).strip()
    low = text.lower()
    for b in DISPLAY_BRANDS:
        if low.startswith(b + " "):
            text = text[len(b):].strip()
            break
    return text[:1].upper() + text[1:]


def short_food_name(name):
    """'Chicken breast, cooked' -> 'Chicken breast' (for meal titles)."""
    return display_food_name(name).split(",")[0].strip()


class NutriEngine:
    def __init__(self, df=None):
        raw = df if df is not None else pd.read_csv(DATA)
        # ---- dedup ----
        self.n_raw = len(raw)
        raw = raw.drop_duplicates(subset=["name"]).reset_index(drop=True)
        self.n_dedup = len(raw)
        for c in NUTRIENTS + ["glycemic_index"]:
            if c in raw: raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
        for c in ["allergens","diet_tags","gerd_trigger","fodmap_flags"]:
            raw[c] = raw[c].fillna("")
        raw["_base_food"] = raw["name"].apply(_canonical_food_name)
        self.df = raw
        # Nutrient direction hints are used only to build the FAISS ideal vector.
        # The main recommender uses the DNN scorer below; these signs only help
        # construct the FAISS baseline query vector.
        self.weights = {"protein_g":1.0,"fiber_g":1.0,"sugar_g":-1.0,"sodium_mg":-1.0,
                        "saturated_fat_g":-0.7,"potassium_mg":0.4,"iron_mg":0.4,
                        "calcium_mg":0.4,"vitamin_b12_ug":0.3,"zinc_mg":0.3}
        self.dnn_feedback = {n: 0.0 for n in NUTRIENTS}
        self._faiss = None; self._faiss_vecs = None
        # RL bandit over food groups for adaptive meal-group selection
        self.bandit = ThompsonBandit(sorted(self.df["food_group"].unique()))

    # ---------- STAGE: clinical filtering ----------
    def clinical_filter(self, df, persona, log):
        keep = pd.Series(True, index=df.index)

        if "IBS" in persona.conditions:
            fodmap = df["fodmap"].astype(str).str.upper()
            names = df["name"].astype(str).str.lower()
            flags = df["fodmap_flags"].astype(str).str.lower()
            allergens = df["allergens"].astype(str).str.lower()

            # IBS: remove high-FODMAP foods plus the most common triggers
            # (garlic, onion, wheat), even when the FODMAP flag misses them.
            IBS_TRIGGERS = ["garlic", "onion", "wheat"]

            m = (fodmap == "H") & keep

            for t in IBS_TRIGGERS:
                m |= names.str.contains(t, na=False) & keep
                m |= flags.str.contains(t, na=False) & keep
                m |= allergens.str.contains(t, na=False) & keep

            for i in df.index[m]:
                why = []
                food_text = (
                    str(df.at[i, "name"]).lower()
                    + " "
                    + str(df.at[i, "fodmap_flags"]).lower()
                    + " "
                    + str(df.at[i, "allergens"]).lower()
                )

                for t in IBS_TRIGGERS:
                    if t in food_text:
                        why.append(t)

                log.clinical[df.at[i, "name"]] = (
                    f"High-FODMAP trigger for IBS ({', '.join(why) or 'high-FODMAP'})"
                )

            keep &= ~m

        if "GERD" in persona.conditions:
            trig = df["gerd_trigger"].astype(str).str.lower()
            names = df["name"].astype(str).str.lower()

            m = pd.Series(False, index=df.index)

            for t in GERD_TRIGGERS:
                t = t.lower()
                m |= trig.str.contains(t, na=False)
                m |= names.str.contains(t, na=False)

            m |= df["fat_g"] > 20      # fried/high-fat reflux trigger
            m &= keep

            for i in df.index[m]:
                why = []
                food_name = str(df.at[i, "name"]).lower()
                food_trig = str(df.at[i, "gerd_trigger"]).lower()

                for t in GERD_TRIGGERS:
                    if t.lower() in food_name or t.lower() in food_trig:
                        why.append(t)

                log.clinical[df.at[i, "name"]] = (
                    f"GERD trigger ({', '.join(why) or 'high-fat'})"
                )

            keep &= ~m

        if "T2_Diabetes" in persona.conditions or persona.gi_max:
            gi_cap = persona.gi_max or 55
            m = (df["glycemic_index"] > gi_cap) & keep
            for i in df.index[m]:
                log.clinical[df.at[i, "name"]] = (
                    f"GI {df.at[i, 'glycemic_index']:.0f} > {gi_cap} (T2 diabetes)"
                )
            keep &= ~m

        if "Hypertension" in persona.conditions:
            m = (df["sodium_mg"] > 120) & keep    # per-100g high-sodium screen (DASH)
            for i in df.index[m]:
                log.clinical[df.at[i, "name"]] = (
                    f"High sodium {df.at[i, 'sodium_mg']:.0f}mg/100g (DASH/hypertension)"
                )
            keep &= ~m

        return df[keep]

    # ---------- STAGE: allergy exclusion (exact set membership) ----------
    def allergy_filter(self, df, allergens, log, lactose_free=False, gluten_free=False):
        excl = set(allergens)
        if gluten_free: excl |= {"gluten","wheat"}
        
        # Oats are often cross-contaminated with gluten, so remove them ONLY for
        # gluten-free users. (Bug fix: this used to run for everyone, which
        # silently removed every oat product from all plans.)
        if gluten_free:
            CELIAC_OAT_EXCLUDE = ["oat", "oats", "oat milk"]
            oat_name = df["name"].astype(str).str.lower()
            oat_base = df.get("_base_food", pd.Series("", index=df.index)).astype(str).str.lower()
            oat_mask = pd.Series(False, index=df.index)
            for term in CELIAC_OAT_EXCLUDE:
                oat_mask |= oat_name.str.contains(term, na=False)
                oat_mask |= oat_base.str.contains(term, na=False)
            for i in df.index[oat_mask]:
                log.allergy[df.at[i, "name"]] = (
                    "Gluten-free: oats are only safe with certified gluten-free labeling"
                )
            df = df[~oat_mask]

        if not excl and not lactose_free:
            return df

        keep = pd.Series(True, index=df.index)
        allerg_series = df["allergens"].astype(str)

        # Lactose-free users get zero dairy: milk, lactose-free milk, cheese,
        # yogurt, butter, cream, etc. Plant milks (soy, almond, oat) and peanut
        # butter are NOT dairy and stay in. (Bug fix: they used to be removed too.)
        dairy_mask = pd.Series(False, index=df.index)
        if lactose_free:
            names = df["name"].astype(str).str.lower()
            allergens_lower = df["allergens"].astype(str).str.lower()
            dairy_words = ["milk", "cheese", "yogurt", "butter", "cream", "whey", "casein"]

            for word in dairy_words:
                dairy_mask |= names.str.contains(word, na=False)
                dairy_mask |= allergens_lower.str.contains(word, na=False)
            not_dairy = names.str.contains(r"\b(?:soy|almond|oat|coconut|rice) milk|peanut butter",
                                           na=False, regex=True)
            dairy_mask &= ~not_dairy

            for i in df.index[dairy_mask]:
                log.allergy[df.at[i, "name"]] = "Excluded dairy product for zero-dairy/lactose-free requirement"

            keep &= ~dairy_mask

        for i in df.index:
            fa = {x for x in allerg_series[i].split("|") if x}
            hit = fa & excl
            lact = lactose_free and str(df.at[i,"lactose"]).upper()=="Y"

            if hit or lact:
                reasons=[]
                if hit:
                    reasons.append("allergen(s): "+", ".join(sorted(hit)))
                if lact:
                    reasons.append("lactose/dairy intolerance")
                if gluten_free and ("gluten" in fa or "wheat" in fa):
                    reasons.append("cross-contamination risk (celiac)")

                log.allergy[df.at[i,"name"]]="; ".join(reasons)
                keep[i]=False

        return df[keep]

    # ---------- STAGE: diet preference ----------
    def diet_filter(self, df, persona, log):
        diet, tags = persona.diet, df["diet_tags"].astype(str)
        if diet=="vegan":          ok = tags.str.contains("vegan")
        elif diet=="vegetarian":   ok = tags.str.contains("vegetarian")
        elif diet=="pescatarian":  ok = tags.str.contains("pescatarian")
        else:                      ok = pd.Series(True, index=df.index)  # nonveg
        if persona.no_pork:
            ok &= ~tags.str.contains("pork")
        for i in df.index[~ok]:
            log.diet[df.at[i,"name"]]=f"Not compatible with {diet} diet" + \
                (" / no pork" if persona.no_pork else "")
        return df[ok]

    # ---------- candidate generation ----------
    def candidates(self, persona):
        log = ExclusionLog(); df = self.df.copy()
        df = self.clinical_filter(df, persona, log)
        df = self.allergy_filter(df, persona.allergens, log,
                        persona.lactose_free, persona.gluten_free)
        df = self.diet_filter(df, persona, log)
        return df.reset_index(drop=True), log

    # ---------- ranking ----------
    def _feature_matrix(self, df):
        X = df[NUTRIENTS].to_numpy(float)
        mu, sd = X.mean(0), X.std(0)+1e-9
        return ((X-mu)/sd).astype("float32")

    def rank_faiss(self, df, persona):
        """T1: embed foods, build an ideal nutrient vector, FAISS NN search."""
        X = self._feature_matrix(df)
        ideal = np.zeros(X.shape[1], dtype="float32")
        idx = {n:i for i,n in enumerate(NUTRIENTS)}
        for n,w in self.weights.items():
            if n in idx: ideal[idx[n]] = np.sign(w)*2.0
        if HAVE_FAISS:
            index = faiss.IndexFlatL2(X.shape[1]); index.add(X)
            D,_ = index.search(ideal.reshape(1,-1), len(df))
            # distance -> score (closer = better)
            order = np.argsort(D[0])
            score = np.zeros(len(df)); score[order] = np.linspace(1,0,len(df))
        else:
            d = np.linalg.norm(X-ideal,axis=1); score = 1-(d-d.min())/(d.max()-d.min()+1e-9)
        out = df.copy(); out["score"]=score
        return out.sort_values("score",ascending=False).reset_index(drop=True)

    def _minmax(self, df, cols):
        X = df[cols].to_numpy(float)
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        return (X - lo) / (hi - lo + 1e-9)

    def _dnn_context(self, persona):
        """Persona/context vector used by the DNN scorer."""
        return np.array([
            1.0 if "T2_Diabetes" in persona.conditions else 0.0,
            1.0 if "Hypertension" in persona.conditions else 0.0,
            1.0 if "GERD" in persona.conditions else 0.0,
            1.0 if "IBS" in persona.conditions else 0.0,
            1.0 if persona.diet == "vegan" else 0.0,
            1.0 if persona.diet == "vegetarian" else 0.0,
            1.0 if persona.diet == "pescatarian" else 0.0,
            min(max(persona.kcal_target, 1200), 2600) / 2600.0,
        ], dtype="float32")

    def _dnn_training_labels(self, df, persona):
        """Create supervised labels for the DNN from persona nutrition goals.

        The app does not have historical dietitian ratings, so we generate weak labels
        from clinical nutrition objectives after all unsafe foods have already been
        filtered out. The neural network then learns a non-linear ranking function over
        nutrients + persona context instead of directly sorting by hand-written weights.
        """
        nutrient_cols = NUTRIENTS + ["glycemic_index"]
        X = self._minmax(df, nutrient_cols)
        col = {n: i for i, n in enumerate(nutrient_cols)}

        y = np.zeros(len(df), dtype="float32")
        # General quality targets.
        y += 1.20 * X[:, col["protein_g"]]
        y += 1.25 * X[:, col["fiber_g"]]
        y += 0.60 * X[:, col["potassium_mg"]]
        y += 0.45 * X[:, col["iron_mg"]]
        y += 0.40 * X[:, col["calcium_mg"]]
        y += 0.35 * X[:, col["vitamin_b12_ug"]]
        y += 0.35 * X[:, col["vitamin_d_ug"]]
        y += 0.30 * X[:, col["zinc_mg"]]
        y -= 1.20 * X[:, col["sugar_g"]]
        y -= 1.10 * X[:, col["sodium_mg"]]
        y -= 0.90 * X[:, col["saturated_fat_g"]]

        # Persona-specific weak labels.
        if "T2_Diabetes" in persona.conditions:
            y += 0.95 * X[:, col["fiber_g"]]
            y -= 1.25 * X[:, col["sugar_g"]]
            y -= 1.10 * X[:, col["glycemic_index"]]
        if "Hypertension" in persona.conditions:
            y += 1.10 * X[:, col["potassium_mg"]]
            y -= 1.35 * X[:, col["sodium_mg"]]
        if "GERD" in persona.conditions:
            y -= 0.90 * X[:, col["fat_g"]]
            y -= 0.85 * X[:, col["saturated_fat_g"]]
        if persona.diet in {"vegan", "vegetarian"}:
            y += 0.65 * X[:, col["iron_mg"]]
            y += 0.45 * X[:, col["calcium_mg"]]
        if persona.diet == "pescatarian":
            y += 0.70 * X[:, col["vitamin_b12_ug"]]
            y += 0.45 * X[:, col["zinc_mg"]]
        for m in persona.micro_priority:
            if m in col:
                y += 0.90 * X[:, col[m]]

        # Online thumbs-up/down feedback nudges weak labels before DNN training.
        for n, delta in self.dnn_feedback.items():
            if n in col:
                y += float(delta) * X[:, col[n]]

        return (y - y.min()) / (y.max() - y.min() + 1e-9)

    def rank_dnn(self, df, persona):
        """T2: supervised DNN scorer for persona-aware food ranking.

        This is a real neural-network ranking model using scikit-learn's
        MLPRegressor. For each persona, the pipeline creates weak supervision labels
        from nutrition/clinical goals, trains a two-hidden-layer feed-forward DNN on
        the candidate food pool, and uses model.predict() as the ranking score.
        Inputs = scaled nutrients + persona context + micronutrient priority features.
        """
        if len(df) == 0:
            out = df.copy(); out["score"] = []
            return out

        nutrient_cols = NUTRIENTS + ["glycemic_index"]
        X_food = df[nutrient_cols].to_numpy(float).astype("float32")
        ctx = np.tile(self._dnn_context(persona), (len(df), 1)).astype("float32")

        # Priority-feature block lets the DNN learn interactions between the user's
        # micronutrient priorities and the actual nutrient amounts in each food.
        priority = np.zeros_like(X_food, dtype="float32")
        col = {n: i for i, n in enumerate(nutrient_cols)}
        for m in persona.micro_priority:
            if m in col:
                priority[:, col[m]] = X_food[:, col[m]]

        features = np.hstack([X_food, ctx, priority]).astype("float32")
        labels = self._dnn_training_labels(df, persona)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Two hidden layers make this a DNN rather than a plain linear scorer.
        # The model is intentionally small so Streamlit generation stays <60 seconds.
        dnn = MLPRegressor(
            hidden_layer_sizes=(48, 16),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=0.01,
            max_iter=180,
            random_state=423,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.15,
        )
        dnn.fit(X_scaled, labels)
        score = dnn.predict(X_scaled).astype("float32")
        score = (score - score.min()) / (score.max() - score.min() + 1e-9)

        out = df.copy()
        out["score"] = score
        return out.sort_values("score", ascending=False).reset_index(drop=True)

    def rank(self, df, persona, technique="dnn"):
        return self.rank_faiss(df, persona) if technique == "faiss" else self.rank_dnn(df, persona)

    # ---------- diversity (MMR) ----------
    def diversify(self, ranked, k=40, lambda_=0.7, pool_n=200):
        top=ranked.head(pool_n).reset_index(drop=True)
        if len(top)<=k: return top
        feats=top[NUTRIENTS].to_numpy(float)
        feats=(feats-feats.mean(0))/(feats.std(0)+1e-9)
        sims=cosine_similarity(feats); rel=top["score"].to_numpy(float)
        sel=[0]; chosen={0}; max_sim=sims[0].copy()
        while len(sel)<k:
            mmr=lambda_*rel-(1-lambda_)*max_sim; 
            for c in chosen: mmr[c]=-1e9
            nxt=int(np.argmax(mmr)); sel.append(nxt); chosen.add(nxt)
            max_sim=np.maximum(max_sim,sims[nxt])
        return top.iloc[sel].reset_index(drop=True)

    # ---------- diversity score ----------
    @staticmethod
    def diversity_score(plan):
        """How varied the week is (0..1). Combines:
        1) base-food uniqueness,
        2) food-group coverage, and
        3) day-level category spread.

        Higher means less repetition across the week.
        """
        foods = [
            f
            for d in plan
            for meal in d["meals"].values()
            for f in meal
        ]
        if not foods:
            return 0.0

        names = [
            f.get("_base_food", _canonical_food_name(f.get("name", "")))
            for f in foods
        ]
        groups = [str(f.get("food_group", "")) for f in foods]

        uniq = len(set(names)) / max(len(names), 1)
        group_coverage = len(set(groups)) / 6.0

        # Daily spread: each day should include several different food groups,
        # not the same category pattern repeated all week.
        daily_spreads = []
        for d in plan:
            day_groups = {
                str(f.get("food_group", ""))
                for meal in d["meals"].values()
                for f in meal
            }
            daily_spreads.append(min(len(day_groups) / 5.0, 1.0))
        daily_spread = sum(daily_spreads) / max(len(daily_spreads), 1)

        score = (
            0.15 * uniq
            + 0.45 * min(group_coverage, 1.0)
            + 0.40 * min(daily_spread, 1.0)
        )
        return round(min(score, 1.0), 3)

    # ---------- meal assignment: 7 days x 3 realistic meals ----------
    def build_plan(self, persona, technique="dnn", seed=None, avoid=None, prefer=None,
                   locked_plan=None, regen=None, **_ignored):
        """Build a 7-day plan of realistic meals.

        Each meal follows a template (e.g. protein + starch + vegetable), every
        food gets a realistic serving size, and the ranking model (DNN or FAISS)
        decides which foods fill each slot.

        avoid       : base foods the user skipped (left out of this plan)
        prefer      : base foods the user saved (favored in this plan)
        locked_plan : an existing plan to keep, except for the meals in `regen`
        regen       : set of (day, meal) pairs to rebuild, e.g. {(3, "Lunch")}
        """
        rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        avoid, prefer, regen = set(avoid or []), set(prefer or []), set(regen or [])

        cand, log = self.candidates(persona)
        if len(cand) == 0:
            return None, log, None
        if avoid:
            skipped = cand["_base_food"].isin(avoid)
            if (~skipped).any():
                for nm in cand.loc[skipped].drop_duplicates("_base_food")["name"]:
                    log.skipped[short_food_name(nm)] = "You skipped a meal with this food"
                cand = cand[~skipped].reset_index(drop=True)

        ranked = self.rank(cand, persona, technique)

        # One row per base food: the best-scoring variant of each food.
        best = (ranked.sort_values("score", ascending=False)
                      .drop_duplicates("_base_food").reset_index(drop=True))
        best["_s"] = best["score"].rank(pct=True)
        foods = {}
        for _, r in best.iterrows():
            base = r["_base_food"]; kcal = max(float(r["energy_kcal"]), 1.0)
            foods[base] = {
                "row": r, "kcal": kcal, "score": float(r["_s"]),
                "group": str(r.get("food_group", "")),
                "roles": roles_for(base, r.get("food_group", ""), kcal),
                "serving": serving_range(base, r.get("food_group", ""), kcal),
                "fish": str(r.get("fish_seafood", "N")).upper() == "Y",
            }

        week_count, fish_count = {}, 0

        def register(f):
            nonlocal fish_count
            b = f["_base_food"]
            week_count[b] = week_count.get(b, 0) + 1
            if str(f.get("fish_seafood", "N")).upper() == "Y":
                fish_count += 1

        def portion(b, kcal_target):
            lo, hi = foods[b]["serving"]
            g = kcal_target / foods[b]["kcal"] * 100.0
            return snap_portion(b, min(max(g, lo), hi), lo, hi)

        def pick(role, kcal_target, day_used, allowed=None, fish=False):
            if allowed is not None:
                opts = [b for b in allowed if b in foods and b not in day_used]
            else:
                opts = [b for b, f in foods.items() if role in f["roles"] and b not in day_used]
            if fish:
                fish_opts = [b for b in opts if foods[b]["fish"]]
                opts = fish_opts or opts
            if not opts:
                return None
            best_b, best_g, best_cost = None, None, 1e18
            for b in opts:
                f = foods[b]
                g = portion(b, kcal_target)
                kcal = f["kcal"] * g / 100.0
                c = week_count.get(b, 0)
                cost = abs(kcal - kcal_target) / max(kcal_target, 1.0)   # calorie fit
                cost += 0.6 * (1.0 - f["score"])                          # model ranking
                cost += 0.45 * c                                         # weekly variety
                if c >= REPEAT_CAP.get(role or "bfast_side", 2):
                    cost += 3.0                                          # hard-ish repeat cap
                if b in prefer:
                    cost -= 0.35                                         # user saved it
                cost -= 0.1 * self.bandit.sample(f["group"], rng)        # RL feedback
                cost += 0.25 * rng.random()                              # variety between plans
                if cost < best_cost:
                    best_b, best_g, best_cost = b, g, cost
            return best_b, best_g

        def make_item(b, grams, role):
            f = foods[b]["row"].to_dict()
            f["portion_g"] = int(round(grams))
            f["_scale"] = grams / 100.0
            f["_base_food"] = b
            f["_role"] = role
            return f

        def set_grams(f, g):
            f["portion_g"] = int(round(g)); f["_scale"] = g / 100.0

        def kcal_of(items):
            return sum(float(f.get("energy_kcal", 0)) * float(f.get("_scale", 1)) for f in items)

        def sodium_of(items):
            return sum(float(f.get("sodium_mg", 0)) * float(f.get("_scale", 1)) for f in items)

        def scale_toward(items, target):
            """Nudge portions toward a calorie target, staying inside serving sizes."""
            for _ in range(3):
                cur = kcal_of(items)
                if cur <= 0 or abs(cur - target) / target < 0.03:
                    break
                factor = target / cur
                for f in items:
                    b = f["_base_food"]
                    lo, hi = foods[b]["serving"] if b in foods else (f["portion_g"], f["portion_g"])
                    g = min(max(f["portion_g"] * factor, lo), hi)
                    set_grams(f, snap_portion(b, g, lo, hi))

        def build_meal(meal, day_used, fish_dinner=False):
            target = persona.kcal_target * MEAL_SHARE[meal]
            items, main_base = [], None
            for role, share in MEAL_TEMPLATES[meal]:
                kc = target * share
                if role == "bfast_side":
                    allowed = BFAST_SIDES.get(main_base, DEFAULT_BFAST_SIDES)
                    choice = pick("bfast_side", kc, day_used, allowed=allowed)
                else:
                    choice = pick(role, kc, day_used, fish=(fish_dinner and role == "protein"))
                if choice is None:
                    continue
                b, g = choice
                if role == "bfast_main":
                    main_base = b
                f = make_item(b, g, role)
                items.append(f); day_used.add(b); register(f)
            return items

        def add_snack(items, day_used):
            """If a day is short on calories, add a realistic snack (not bigger portions)."""
            snack = []
            fruit_snacks = ROLE_FOODS["fruit"]
            for _ in range(2):
                short = persona.kcal_target - kcal_of([f for fs in items.values() for f in fs] + snack)
                if short < 120:
                    break
                allowed = None
                if snack:
                    # A second snack item pairs with the first: fruit + something filling
                    # (apple + peanut butter, yogurt + banana), never two of the same kind.
                    first_is_fruit = snack[0]["_base_food"] in fruit_snacks
                    allowed = [b for b in ROLE_FOODS["snack"]
                               if (b in fruit_snacks) != first_is_fruit]
                choice = pick("snack", min(short, 350), day_used, allowed=allowed)
                if choice is None:
                    break
                b, g = choice
                f = make_item(b, g, "snack")
                snack.append(f); day_used.add(b); register(f)
            if snack:
                items["Snack"] = snack

        # Fish meals (for anyone whose diet allows fish)
        fish_days = set()
        if persona.require_fish_meals > 0 and persona.diet in ("nonveg", "pescatarian"):
            fish_days = set(rng.sample(range(1, 8), min(persona.require_fish_meals, 7)))

        locked = {d["day"]: d for d in (locked_plan or [])}
        # Count the meals we're keeping first, so new meals respect weekly variety.
        for dnum, d in locked.items():
            for m, fs in d["meals"].items():
                if (dnum, m) not in regen:
                    for f in fs:
                        register(f)

        plan = []
        for day in range(1, 8):
            if day in locked:
                day_used = {f["_base_food"] for m, fs in locked[day]["meals"].items()
                            if (day, m) not in regen for f in fs}
                items = {}
                for m in MEALS + ["Snack"]:
                    old = locked[day]["meals"].get(m)
                    if (day, m) in regen and m in MEALS:
                        was_fish = any(str(f.get("fish_seafood", "N")).upper() == "Y"
                                       for f in (old or []))
                        items[m] = build_meal(m, day_used, fish_dinner=(m == "Dinner" and was_fish))
                        scale_toward(items[m], persona.kcal_target * MEAL_SHARE[m])
                    elif old is not None:
                        items[m] = [dict(f) for f in old]
                plan.append({"day": day, "meals": items})
                continue

            day_used, items = set(), {}
            for m in MEALS:
                items[m] = build_meal(m, day_used, fish_dinner=(m == "Dinner" and day in fish_days))
            all_items = [f for fs in items.values() for f in fs]
            scale_toward(all_items, persona.kcal_target)
            add_snack(items, day_used)
            scale_toward([f for fs in items.values() for f in fs], persona.kcal_target)

            # Respect a daily sodium cap by trimming portions (never below a serving minimum).
            if persona.sodium_daily_cap:
                all_items = [f for fs in items.values() for f in fs]
                na = sodium_of(all_items)
                if na > persona.sodium_daily_cap:
                    ratio = persona.sodium_daily_cap / na
                    for f in all_items:
                        b = f["_base_food"]; lo, hi = foods[b]["serving"]
                        set_grams(f, snap_portion(b, max(f["portion_g"] * ratio, lo), lo, hi))
            plan.append({"day": day, "meals": items})

        analysis = self.analyze(plan, persona)
        analysis["diversity_score"] = self.diversity_score(plan)
        analysis["fish_meals"] = sum(
            1 for d in plan for m, fs in d["meals"].items()
            if m in ("Lunch", "Dinner") and any(str(f.get("fish_seafood", "N")).upper() == "Y" for f in fs))
        analysis["dedup"] = {"raw": self.n_raw, "after": self.n_dedup}
        return plan, log, analysis

    # ---------- macro/micro analysis vs RDA ----------
    def analyze(self, plan, persona):
        per_day=[]
        for d in plan:
            t={n:0.0 for n in NUTRIENTS}
            for meal,foods in d["meals"].items():
                for f in foods:
                    s=float(f.get("_scale",1.0))
                    for n in NUTRIENTS: t[n]+=float(f.get(n,0))*s
            per_day.append(t)
        days=len(per_day)
        daily={n:sum(d[n] for d in per_day)/days for n in NUTRIENTS}
        rda=rda_for(persona.age, persona.sex)
        # per-day RDA flags (<80%) for tracked micros + fiber
        flags=[]
        tracked=MICROS+["fiber_g","potassium_mg"]
        for di,d in enumerate(per_day,1):
            for n in tracked:
                if n in rda and rda[n]>0 and d[n] < 0.8*rda[n]:
                    flags.append({"day":di,"nutrient":n,
                                  "got":round(d[n],1),"target":rda[n],
                                  "pct":round(100*d[n]/rda[n])})
        return {"daily_avg":daily,"per_day":per_day,"rda":rda,
                "targets":{"energy_kcal":persona.kcal_target},"rda_flags":flags}

    # ---------- adaptive learning ----------
    def apply_feedback(self, food, liked, lr=0.08):
        """Adaptive learning: update the DNN feedback layer and RL bandit arm."""
        d = 1.0 if liked else -1.0
        for n in self.dnn_feedback:
            scale = self.df[n].max() or 1.0
            self.dnn_feedback[n] += lr * d * (float(food.get(n, 0)) / scale - 0.5)
        self.bandit.update(food.get("food_group", "Other"), 1.0 if liked else 0.0)
        return {"dnn_feedback": dict(self.dnn_feedback), "bandit": self.bandit.means()}

    # The app shares one cached engine across all visitors, so each visitor's
    # feedback is saved in their own session and loaded before each plan.
    def feedback_state(self):
        return {"dnn_feedback": dict(self.dnn_feedback),
                "alpha": dict(self.bandit.alpha), "beta": dict(self.bandit.beta)}

    def load_feedback_state(self, state=None):
        self.dnn_feedback = {n: 0.0 for n in NUTRIENTS}
        self.bandit = ThompsonBandit(sorted(self.df["food_group"].unique()))
        if state:
            self.dnn_feedback.update(state.get("dnn_feedback", {}))
            self.bandit.alpha.update(state.get("alpha", {}))
            self.bandit.beta.update(state.get("beta", {}))


def default_personas():
    return [
        Persona("Priya - IBS + Vegetarian + Lactose-intolerant",
                ["IBS"], [], "vegetarian", 1800, age=32, sex="female",
                lactose_free=True, micro_priority=["iron_mg","calcium_mg","vitamin_d_ug"]),
        Persona("Ravi - GERD + Non-veg + Gluten-free",
                ["GERD"], [], "nonveg", 2200, age=40, sex="male",
                no_pork=True, gluten_free=True,
                micro_priority=["vitamin_b12_ug","zinc_mg","magnesium_mg"]),
        Persona("Mei - T2 Diabetes + Vegan + Tree-nut allergy",
                ["T2_Diabetes"], ["tree_nut"], "vegan", 1600, age=55, sex="female",
                gi_max=55, micro_priority=["vitamin_b12_ug","iron_mg","zinc_mg"]),
        Persona("James - Hypertension + Pescatarian + Soy allergy",
                ["Hypertension"], ["soy"], "pescatarian", 2000, age=48, sex="male",
                sodium_daily_cap=1500, require_fish_meals=3,
                micro_priority=["potassium_mg","magnesium_mg"]),
    ]
