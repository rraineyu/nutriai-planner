"""
app.py - NutriAI: personalized 7-day meal plan builder
Builds a meal plan around a user's health conditions, allergies, and diet
preferences, with nutrition analysis, explainable exclusions, adaptive
feedback, and PDF/CSV export.
Run: streamlit run code/app.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import streamlit as st
from engine import (NutriEngine, Persona, NUTRIENTS, MACROS,
                    MICROS, rda_for, ALLERGEN_CHOICES,
                    display_food_name, short_food_name, portion_label)
from export import plan_to_csv, plan_to_pdf

st.set_page_config(page_title="NutriAI - Diet Plan Builder", page_icon="🥗", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Nunito:wght@400;500;600;700&display=swap');

:root{
  --cream:#fcf8f3; --cream2:#f6efe6;
  --sage:#a8c8a0; --sage-deep:#6fa07e; --sage-bg:#e8f1e4;
  --peach:#f6c9a8; --peach-bg:#fbe6d6; --peach-deep:#d98e5f;
  --lavender:#cdc1e8; --lavender-bg:#ece7f7;
  --butter:#f6e2a0; --butter-bg:#fbf3d4;
  --sky:#bcd6e8; --sky-bg:#e4f0f7;
  --ink:#4a4034; --ink-soft:#7a6f60; --line:#ece2d2;
}
html, body, [class*="css"] { font-family:'Nunito',system-ui,sans-serif; color:var(--ink); }
h1,h2,h3,h4 { font-family:'Fraunces',Georgia,serif !important; letter-spacing:-0.01em; color:var(--ink); }
.stApp { background:
   radial-gradient(900px 500px at 12% -8%, var(--peach-bg) 0%, transparent 55%),
   radial-gradient(800px 480px at 92% 4%, var(--lavender-bg) 0%, transparent 50%),
   radial-gradient(700px 600px at 70% 100%, var(--sky-bg) 0%, transparent 55%),
   linear-gradient(180deg,var(--cream) 0%, var(--cream2) 100%);
   background-attachment:fixed; }
.block-container { padding-top:3.6rem; max-width:1180px; }
/* Streamlit's top bar: transparent so it never paints over the page */
header[data-testid="stHeader"]{ background:transparent; }

/* big title */
h1 { font-size:2.7rem !important; }

/* metric cards - soft pastel tints, rounded, gentle shadow */
.metric-card{
  background:linear-gradient(160deg,#ffffff 0%, var(--sage-bg) 130%);
  border:1px solid #e3ecdd; border-radius:20px; padding:16px 18px;
  box-shadow:0 6px 18px -10px rgba(111,160,126,.45); }

/* meal cards - friendly rounded, colored left accent, hover lift */
.meal-card{
  background:#fffdfa; border:1px solid var(--line); border-left:5px solid var(--sage);
  border-radius:16px; padding:13px 16px; margin-bottom:6px;
  box-shadow:0 4px 14px -10px rgba(74,64,52,.35);
  transition:transform .15s ease, box-shadow .15s ease; }
.meal-card:hover{ transform:translateY(-2px);
  box-shadow:0 10px 22px -12px rgba(217,142,95,.5); }
.meal-card small{ color:var(--ink-soft); }

/* day chip - pill, pastel */
.daychip{display:inline-block;
  background:linear-gradient(135deg,var(--sage) 0%, var(--sage-deep) 100%);
  color:#fff;border-radius:999px;padding:6px 20px;
  font-family:'Fraunces',serif;font-weight:600;font-size:1.05rem;
  box-shadow:0 6px 16px -8px rgba(111,160,126,.7); margin:6px 0; }

.excl{color:var(--peach-deep);font-size:0.94em;}

/* buttons - rounded pastel, friendly hover */
.stButton>button{
  border-radius:14px; border:1.5px solid var(--sage);
  background:#fff; color:var(--sage-deep); font-weight:600;
  transition:all .15s ease; }
.stButton>button:hover{
  background:var(--sage-bg); border-color:var(--sage-deep); color:var(--ink);
  transform:translateY(-1px); }
/* primary generate button */
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,var(--sage) 0%, var(--sage-deep) 100%);
  color:#fff; border:none;
  box-shadow:0 8px 20px -8px rgba(111,160,126,.8); }
.stButton>button[kind="primary"]:hover{ filter:brightness(1.04); transform:translateY(-1px); }

/* download buttons */
.stDownloadButton>button{
  border-radius:14px; border:1.5px solid var(--peach); background:var(--peach-bg);
  color:var(--peach-deep); font-weight:700; }
.stDownloadButton>button:hover{ background:var(--peach); color:#fff; }

/* tabs - pill style */
.stTabs [data-baseweb="tab-list"]{ gap:6px; }
.stTabs [data-baseweb="tab"]{
  background:#fff; border:1px solid var(--line); border-radius:999px;
  padding:6px 16px; font-weight:600; color:var(--ink-soft); }
.stTabs [aria-selected="true"]{
  background:var(--sage-bg) !important; color:var(--ink) !important;
  border-color:var(--sage) !important; }

/* sidebar */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#fff 0%, var(--cream2) 100%);
  border-right:1px solid var(--line); }

/* metric native */
[data-testid="stMetricValue"]{ font-family:'Fraunces',serif; color:var(--sage-deep); }
hr{ border-color:var(--line); }

/* ---- page 1: profile form (fits one screen) ---- */
.hero{ text-align:center; margin:0 0 14px; }
.hero h1{ font-size:2.5rem !important; margin:0 !important; padding:0 0 .2rem !important; }
.hero p{ color:var(--ink-soft); font-size:1rem; margin:0; }
.sec-label{ font-size:.8rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:var(--sage-ink); padding-bottom:8px; margin-bottom:4px; border-bottom:1px solid var(--line); }
.form-note{ text-align:center; color:var(--ink-soft); font-size:.82rem; margin-top:8px; }
/* the Create button: your green, not Streamlit red */
.stFormSubmitButton>button{ border-radius:14px; min-height:50px; font-weight:700; font-size:1.05rem;
  background:var(--sage-ink) !important; border:none !important; color:#fff !important; }
.stFormSubmitButton>button:hover{ background:var(--sage-ink-dark) !important; }
/* sliders and selected pills in green too */
[data-testid="stSlider"] [role="slider"]{ background:var(--sage-ink) !important; }
[data-testid="stPills"] button[kind="pillsActive"]{ background:var(--sage-bg) !important;
  border-color:var(--sage-ink) !important; color:var(--sage-ink-dark) !important; }
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{ display:none; }
/* ---- page 2: plan header ---- */
.plan-brand{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:2rem; color:var(--ink); }
.plan-brand span{ font-family:'Nunito',sans-serif; font-weight:600; font-size:1.05rem; color:var(--ink-soft); }
/* ---- consolidated plan layout ---- */
:root{ --sage-ink:#3F7652; --sage-ink-dark:#2E5A3D; }
.summary-bar{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between;
  gap:10px 24px; background:#fff; border:1px solid var(--line); border-radius:18px;
  padding:14px 20px; margin:4px 0 14px; }
.summary-bar .stats{ display:flex; flex-wrap:wrap; align-items:center; gap:6px 22px; }
.summary-bar b{ font-size:1.15rem; }
.summary-bar .muted{ color:var(--ink-soft); }
.flag{ padding:4px 10px; border-radius:999px; background:var(--peach-bg); color:#6B3A1A;
  font-weight:700; font-size:.82rem; }
.profile-note{ color:var(--ink-soft); font-size:.9rem; }
.day-head{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:baseline;
  gap:6px; margin:8px 0 2px; }
.day-head h2{ margin:0 !important; padding:0 !important; font-size:1.8rem !important; }
.day-head span{ color:var(--ink-soft); font-size:.95rem; }
.meal-tag{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:700; }
.meal-tag.Breakfast{ background:var(--peach-bg); color:#8A4A22; }
.meal-tag.Lunch{ background:var(--sage-bg); color:var(--sage-ink-dark); }
.meal-tag.Dinner{ background:var(--lavender-bg); color:#4B3F7A; }
.meal-title{ font-weight:700; font-size:1.08rem; line-height:1.3; margin:8px 0 4px; min-height:2.6em; }
.meal-meta{ color:var(--ink-soft); font-size:.9rem; margin-bottom:2px; }
.portion-row{ display:flex; justify-content:space-between; gap:10px; padding:3px 0; font-size:.95rem; }
.portion-row span:last-child{ color:var(--ink-soft); white-space:nowrap; }
.snack-strip{ display:flex; flex-wrap:wrap; align-items:center; gap:6px 14px; padding:12px 16px;
  border-radius:14px; background:#FCF8E8; border:1px solid #F0E6C2; margin-top:6px; }
.snack-strip .meal-tag{ background:#F6E7B0; color:#5C4710; }
.snack-strip .muted{ color:var(--ink-soft); }
/* bordered containers = meal cards */
[data-testid="stVerticalBlockBorderWrapper"]{ background:#fff; border-radius:18px !important; }
/* day picker: equal-width pills, selected one in deep sage */
[data-testid="stButtonGroup"] button{ border-radius:14px !important; min-height:46px; font-weight:700; }
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"]{
  background:var(--sage-ink) !important; color:#fff !important; border-color:var(--sage-ink) !important; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_engine(): return NutriEngine()
# Each visitor keeps their own feedback in their session. The engine is shared
# by everyone, so we load this visitor's feedback into it on every run.
for k,v in {"fb_state":None,"plan":None,"feedback_log":[],
            "skipped_foods":[],"saved_foods":[]}.items():
    if k not in st.session_state: st.session_state[k]=v
eng=get_engine()
eng.load_feedback_state(st.session_state.fb_state)


CONDS=["IBS","GERD","T2_Diabetes","Hypertension"]

TECHNIQUE_LABELS = {
    "dnn": "Best nutrition fit (recommended)",
    "faiss": "Fast match"
}
TECHNIQUE_HELP = {
    "dnn": "Uses NutriAI's neural network to pick foods that best match your nutrition targets.",
    "faiss": "Finds similar foods quickly. Faster, but a looser nutrition fit."
}

# Nutrients where staying UNDER the target is the goal (limits, not minimums)
UPPER_LIMIT_NUTRIENTS = {"sodium_mg"}

UNIT_LABELS = {"mg": "mg", "ug": "µg", "g": "g", "kcal": "kcal"}

def nutrient_label(key):
    """vitamin_b12_ug -> 'Vitamin B12 (µg)'"""
    parts = str(key).split("_")
    unit = UNIT_LABELS.get(parts[-1]) if len(parts) > 1 else None
    words = parts[:-1] if unit else parts
    name = " ".join(w.upper() if (len(w) <= 3 and any(ch.isdigit() for ch in w)) else w.capitalize()
                    for w in words)
    return f"{name} ({unit})" if unit else name

def _clean_food_name(food):
    return display_food_name(food.get("name", ""))

def meal_title(meal, foods):
    """Name the meal after what's in it, e.g. 'Salmon with quinoa & kale'."""
    names = [short_food_name(f.get("name", "")) for f in foods]
    if not names:
        return meal
    first, rest = names[0], [n[:1].lower() + n[1:] for n in names[1:]]
    if not rest:
        return first
    if len(rest) == 1:
        return f"{first} with {rest[0]}"
    return f"{first} with {', '.join(rest[:-1])} & {rest[-1]}"

def plan_fit_summary(persona, analysis):
    """Short personalized summary shown after generation."""
    parts = []
    if persona.get("conditions"):
        readable = {
            "IBS": "IBS-sensitive triggers",
            "GERD": "GERD reflux triggers",
            "T2_Diabetes": "high-glycemic foods",
            "Hypertension": "higher-sodium foods",
        }
        parts.append("avoids " + ", ".join(readable.get(c, c) for c in persona.get("conditions", [])))
    if persona.get("allergens"):
        parts.append("removes " + ", ".join(a.replace("_", " ") for a in persona.get("allergens", [])) + " allergens")
    if persona.get("lactose_free"):
        parts.append("keeps meals lactose-free")
    if persona.get("gluten_free"):
        parts.append("keeps meals gluten-free")
    if persona.get("diet") and persona.get("diet") != "nonveg":
        parts.append(f"stays {persona.get('diet')}")
    kcal = analysis.get("daily_avg", {}).get("energy_kcal", 0)
    base = "This plan " + "; ".join(parts) + "." if parts else "This plan is built around your profile and nutrition targets."
    return f"{base} It averages about {kcal:,.0f} calories a day, with a variety of foods across the week."

def ingredient_line(foods):
    return ", ".join(f"{_clean_food_name(f)} ({portion_label(f)})" for f in foods)

def regenerate_meal(day_num, meal, foods):
    """Swap one meal for a new one, keeping the rest of the week."""
    st.session_state.skipped_foods = sorted(set(st.session_state.skipped_foods)
                                            | {f["_base_food"] for f in foods})
    s = st.session_state.plan
    new_plan, new_log, new_an = eng.build_plan(
        Persona(**s["persona"]), s["technique"], seed=time.time_ns() % (1 << 30),
        avoid=st.session_state.skipped_foods, prefer=st.session_state.saved_foods,
        locked_plan=s["plan"], regen={(day_num, meal)})
    if new_plan is not None:
        s["plan"], s["analysis"] = new_plan, new_an
        s["log"] = new_log.__dict__

# ---------------- Page 1: the profile form ----------------
DIET_LABELS={"nonveg":"No restrictions (eats everything)","pescatarian":"Pescatarian (fish, no meat)",
             "vegetarian":"Vegetarian","vegan":"Vegan"}
ALLERGEN_LABELS={"milk":"Milk","egg":"Egg","fish":"Fish","shellfish":"Shellfish",
                 "tree_nut":"Tree nuts","peanut":"Peanuts","soy":"Soy",
                 "wheat":"Wheat","gluten":"Gluten"}
COND_LABELS={"IBS":"IBS","GERD":"GERD / acid reflux","T2_Diabetes":"Type 2 diabetes",
             "Hypertension":"High blood pressure"}
DEFAULT_PROFILE=dict(age=35,sex="female",kcal=2000,diet="nonveg",allergens=[],lact=False,
                     gf=False,pork=False,conditions=[],na=0,gi=0,fish=0,technique="dnn")
for k,v in {"view":"form","profile":dict(DEFAULT_PROFILE),"plan_error":None}.items():
    if k not in st.session_state: st.session_state[k]=v

def persona_from(pr):
    return Persona(name="Custom profile",conditions=pr["conditions"],allergens=pr["allergens"],
        diet=pr["diet"],kcal_target=pr["kcal"],age=pr["age"],sex=pr["sex"],
        lactose_free=pr["lact"],no_pork=pr["pork"],gluten_free=pr["gf"],
        require_fish_meals=pr["fish"],sodium_daily_cap=pr["na"],gi_max=pr["gi"],micro_priority=[])

def make_plan(pr):
    """Build a plan from a saved profile; go to the plan page if it worked."""
    persona=persona_from(pr)
    t0=time.time()
    plan,log,an=eng.build_plan(persona,pr["technique"],seed=time.time_ns()%(1<<30),
                               avoid=st.session_state.skipped_foods,
                               prefer=st.session_state.saved_foods)
    if plan is None:
        st.session_state.plan_error=("No foods match all of your restrictions. "
                                     "Try removing one restriction and create the plan again.")
        st.session_state.view="form"
        return
    st.session_state.plan={"plan":plan,"log":log.__dict__,"analysis":an,
        "persona":persona.__dict__,"gt":time.time()-t0,"technique":pr["technique"]}
    st.session_state.plan_day="Day 1"
    st.session_state.plan_error=None
    st.session_state.view="plan"

def profile_page():
    """Page 1: the whole form on one screen, three sections side by side."""
    pr=st.session_state.profile
    st.markdown('<div class="hero"><h1>🍽️ NutriAI</h1>'
                '<p>Tell us how you eat. We\'ll build a 7-day meal plan with real portions and a grocery list.</p></div>',
                unsafe_allow_html=True)
    if st.session_state.plan_error:
        st.error(st.session_state.plan_error)
    AVOID_OPTS={"lact":"Lactose-free","gf":"Gluten-free","pork":"No pork"}
    with st.form("profile_form", border=False):
        c1,c2,c3=st.columns(3,gap="medium")
        with c1, st.container(border=True):
            st.markdown('<div class="sec-label">1 · About you</div>',unsafe_allow_html=True)
            a1,a2=st.columns(2)
            age=a1.number_input("Age",10,100,pr["age"])
            sex=a2.selectbox("Sex",["female","male"],index=0 if pr["sex"]=="female" else 1,
                             format_func=str.capitalize)
            kcal=st.slider("Daily calorie goal",1200,3000,pr["kcal"],50,format="%d kcal")
        with c2, st.container(border=True):
            st.markdown('<div class="sec-label">2 · Food preferences</div>',unsafe_allow_html=True)
            diet=st.selectbox("Diet",list(DIET_LABELS),index=list(DIET_LABELS).index(pr["diet"]),
                              format_func=lambda k: DIET_LABELS[k])
            allergens=st.multiselect("Allergies to avoid",ALLERGEN_CHOICES,default=pr["allergens"],
                                     format_func=lambda k: ALLERGEN_LABELS.get(k,k.replace("_"," ").capitalize()),
                                     placeholder="None")
            also=st.pills("Also",list(AVOID_OPTS),selection_mode="multi",
                          default=[k for k in AVOID_OPTS if pr[k]],
                          format_func=lambda k: AVOID_OPTS[k],label_visibility="collapsed")
        with c3, st.container(border=True):
            st.markdown('<div class="sec-label">3 · Health needs</div>',unsafe_allow_html=True)
            conditions=st.multiselect("Conditions",CONDS,default=pr["conditions"],
                                      format_func=lambda k: COND_LABELS.get(k,k),placeholder="None")
            with st.expander("Advanced settings (optional)"):
                st.caption("Use these if a doctor or dietitian has given you specific limits.")
                na=st.slider("Sodium daily cap (mg, 0 = off)",0,2500,pr["na"],50)
                gi=st.slider("Max glycemic index (0 = off)",0,100,pr["gi"],5)
                fish=st.slider("Minimum fish meals per week",0,7,pr["fish"])
                technique=st.radio("Plan style",["dnn","faiss"],
                    index=["dnn","faiss"].index(pr["technique"]),
                    format_func=lambda t: TECHNIQUE_LABELS[t],
                    captions=[TECHNIQUE_HELP["dnn"],TECHNIQUE_HELP["faiss"]])
            st.caption("Sodium and blood-sugar limits, fish meals, plan style.")
        _,mid,_=st.columns([1,1.3,1])
        submitted=mid.form_submit_button("Create my 7-day plan",type="primary",width="stretch")
    if submitted:
        also=set(also or [])
        st.session_state.profile=dict(age=age,sex=sex,kcal=kcal,diet=diet,allergens=allergens,
            lact="lact" in also,gf="gf" in also,pork="pork" in also,conditions=conditions,
            na=na,gi=gi,fish=fish,technique=technique)
        with st.spinner("Building your 7-day plan..."):
            make_plan(st.session_state.profile)
        st.rerun()
    if st.session_state.plan and st.session_state.view=="form":
        _,mid,_=st.columns([1,1.3,1])
        mid.button("← Back to my plan",on_click=lambda: st.session_state.update(view="plan"),
                   width="stretch")
    st.markdown('<p class="form-note">General meal-planning guidance, not medical advice. '
                'If you have a health condition, check your plan with a dietitian or doctor.</p>',
                unsafe_allow_html=True)

if st.session_state.view!="plan" or not st.session_state.plan:
    profile_page()
    st.stop()

def on_save(day_num, meal):
    foods = st.session_state.plan["plan"][day_num-1]["meals"][meal]
    for f in foods:
        eng.apply_feedback(f, True)
    st.session_state.fb_state = eng.feedback_state()
    st.session_state.saved_foods = sorted(set(st.session_state.saved_foods)
                                          | {f["_base_food"] for f in foods})
    title = meal_title(meal, foods)
    st.session_state.feedback_log.append((title, "saved"))
    st.toast(f"Saved: {title}. NutriAI will use these foods more in your next plans.", icon="⭐")

def on_skip(day_num, meal):
    foods = st.session_state.plan["plan"][day_num-1]["meals"][meal]
    for f in foods:
        eng.apply_feedback(f, False)
    st.session_state.fb_state = eng.feedback_state()
    title = meal_title(meal, foods)
    regenerate_meal(day_num, meal, foods)
    new_title = meal_title(meal, st.session_state.plan["plan"][day_num-1]["meals"][meal])
    st.session_state.feedback_log.append((title, "skipped"))
    st.toast(f"Swapped Day {day_num} {meal.lower()}: {new_title}.", icon="🔄")

def go_edit(): st.session_state.view="form"
def go_new_plan(): make_plan(st.session_state.profile)

h1,h2=st.columns([3,2],vertical_alignment="center")
h1.markdown('<div class="plan-brand">🍽️ NutriAI <span>· Your 7-day plan</span></div>',
            unsafe_allow_html=True)
with h2:
    b1,b2=st.columns(2)
    b1.button("✏️ Edit profile",on_click=go_edit,width="stretch")
    b2.button("🔄 New plan",on_click=go_new_plan,width="stretch",
              help="Build a fresh week with the same profile")

tab_plan,tab_an,tab_groc,tab_excl=st.tabs(
    ["My plan 📝 ","Nutrition 📊","Grocery list 🛒","What we left out 🚫"])
state=st.session_state.plan

def _kcal(foods): return sum(float(f["energy_kcal"])*float(f["_scale"]) for f in foods)
def _protein(foods): return sum(float(f["protein_g"])*float(f["_scale"]) for f in foods)

def summary_bar(state):
    """One line instead of four metric cards."""
    an=state["analysis"]; da=an.get("daily_avg",{}); per=state["persona"]
    goal=per.get("kcal_target") or 0
    # Flag the nutrient that falls short on the most days (if it's a pattern, 3+ days)
    days_low={}
    for fl in an.get("rda_flags",[]):
        if fl.get("nutrient") in UPPER_LIMIT_NUTRIENTS: continue
        days_low.setdefault(fl["nutrient"],set()).add(fl["day"])
    flag=""
    if days_low:
        nut,days=max(days_low.items(),key=lambda kv:len(kv[1]))
        if len(days)>=3:
            flag=f'<span class="flag">{nutrient_label(nut).split(" (")[0]} low on {len(days)} of 7 days</span>'
    div=an.get("diversity_score")
    div_txt=f' · variety {float(div):.2f}' if isinstance(div,(int,float)) else ""
    diet=DIET_LABELS.get(per.get("diet"),"").split(" (")[0]
    st.markdown(
        f'<div class="summary-bar"><div class="stats">'
        f'<span><b>{da.get("energy_kcal",0):,.0f}</b> <span class="muted">avg kcal of {goal:,}</span></span>'
        f'<span><b>{da.get("protein_g",0):.0f} g</b> <span class="muted">protein</span></span>'
        f'<span><b>{da.get("fiber_g",0):.0f} g</b> <span class="muted">fiber</span></span>{flag}</div>'
        f'<div class="profile-note">{diet} · {goal:,} kcal{div_txt}</div></div>',
        unsafe_allow_html=True)

def meal_card(day_num, meal, foods, saved):
    with st.container(border=True):
        st.markdown(
            f'<span class="meal-tag {meal}">{meal}</span>'
            f'<div class="meal-title">{meal_title(meal, foods)}</div>'
            f'<div class="meal-meta">{_kcal(foods):.0f} kcal · {_protein(foods):.0f} g protein</div>',
            unsafe_allow_html=True)
        with st.expander("Portions"):
            st.markdown("".join(
                f'<div class="portion-row"><span>{_clean_food_name(f)}</span><span>{portion_label(f)}</span></div>'
                for f in foods), unsafe_allow_html=True)
            is_saved = bool(foods) and all(f["_base_food"] in saved for f in foods)
            b1,b2=st.columns(2)
            b1.button("⭐ Saved" if is_saved else "⭐ Save", key=f"save_d{day_num}_{meal}",
                      help="Use these foods more often in your plans",
                      on_click=on_save, args=(day_num, meal), disabled=is_saved, width="stretch")
            b2.button("🔄 Swap", key=f"swap_d{day_num}_{meal}",
                      help="Replace this meal with a different one",
                      on_click=on_skip, args=(day_num, meal), width="stretch")

with tab_plan:
    if not state:
        st.info("Create a plan to get started.")
    else:
        plan=state["plan"]; an=state["analysis"]
        summary_bar(state)

        day_names=[f"Day {d['day']}" for d in plan]
        if st.session_state.get("plan_day") not in day_names:
            st.session_state.plan_day=day_names[0]
        picked=st.segmented_control("Choose a day",day_names,key="plan_day",
                                    label_visibility="collapsed",width="stretch")
        day_num=int((picked or day_names[0]).split()[1])
        day=plan[day_num-1]
        all_foods=[f for fs in day["meals"].values() for f in fs]
        st.markdown(f'<div class="day-head"><h2>Day {day_num}</h2>'
                    f'<span>{_kcal(all_foods):,.0f} kcal · {_protein(all_foods):.0f} g protein</span></div>',
                    unsafe_allow_html=True)

        saved=set(st.session_state.saved_foods)
        cols=st.columns(3)
        for col,meal in zip(cols,("Breakfast","Lunch","Dinner")):
            if meal in day["meals"]:
                with col: meal_card(day_num, meal, day["meals"][meal], saved)
        snack=day["meals"].get("Snack")
        if snack:
            st.markdown(f'<div class="snack-strip"><span class="meal-tag">Snack</span>'
                        f'<b>{ingredient_line(snack)}</b><span class="muted">{_kcal(snack):.0f} kcal</span></div>',
                        unsafe_allow_html=True)

        st.caption("Open **Portions** to see amounts. **Swap** replaces a meal you don't like; "
                   "**Save** shows those foods more often.")
        st.markdown("")
        csvb=plan_to_csv(plan); pdfb=plan_to_pdf(plan,state["persona"],an)
        e1,e2=st.columns(2)
        e1.download_button("⬇ Download PDF (full week)",pdfb,"nutriai_plan.pdf","application/pdf",width="stretch")
        e2.download_button("⬇ Download CSV",csvb,"nutriai_plan.csv","text/csv",width="stretch")

        if st.session_state.feedback_log:
            with st.expander("🧠 Your feedback"):
                st.caption("Saved foods show up more often in your next plans. "
                           "Swapped meals are replaced and those foods are left out.")
                for nm,act in st.session_state.feedback_log[-12:]: st.write(f"• {act}: {nm}")
                if st.button("Reset my feedback", key="reset_fb"):
                    for k in ("fb_state","feedback_log","skipped_foods","saved_foods"):
                        st.session_state[k] = None if k=="fb_state" else []
                    st.rerun()

with tab_an:
    if not state: st.info("Generate a plan to see nutrient analysis.")
    else:
        an=state["analysis"]; da=an["daily_avg"]; rda=an["rda"]
        st.markdown("### Daily macros")
        cols=st.columns(5)
        for col,m in zip(cols,MACROS):
            label={"energy_kcal":"Calories","protein_g":"Protein g","carb_g":"Carb g",
                   "fat_g":"Fat g","fiber_g":"Fiber g"}[m]
            col.markdown(f'<div class="metric-card"><small>{label}</small><br>'
                f'<span style="font-size:1.5em;font-family:Fraunces">{da[m]:.0f}</span></div>',
                unsafe_allow_html=True)
        st.markdown("### Micronutrients vs RDA (tailored by age/sex)")
        mrows=[]
        
        sex = str(state["persona"].get("sex","")).lower()
        
        target_overrides = {
            "sodium_mg": 1500,
            "potassium_mg": 3400 if state["persona"]["sex"] == "male" else 2600,
            "magnesium_mg": 400 if state["persona"]["sex"] == "male" else 310,
        }
        display_micros = MICROS + ["sodium_mg", "potassium_mg", "magnesium_mg"]
        # If the user set a sodium cap, that cap is their target
        if state["persona"].get("sodium_daily_cap"):
            target_overrides["sodium_mg"] = state["persona"]["sodium_daily_cap"]
        # dict.fromkeys de-dupes in case MICROS already includes these
        for m in dict.fromkeys(display_micros):
            if m not in da:
                continue
            tgt = target_overrides.get(m, rda.get(m, 0))
            pct = 100 * da[m] / tgt if tgt else 0
            if m in UPPER_LIMIT_NUTRIENTS:
                # Limits: under target is good, over is the warning
                status = "✅ Under limit" if da[m] <= tgt else "⚠️ Over limit"
            else:
                status = "✅ Met" if pct >= 80 else "⚠️ Below 80%"
            # (Bug fix: the old code computed `status` above, then ignored it here)
            mrows.append({"Nutrient":nutrient_label(m),"Daily avg":round(da[m],1),
                          "Target":tgt,"% of target":round(pct),"Status":status})
        st.dataframe(pd.DataFrame(mrows),width="stretch",hide_index=True)
        # Limit nutrients don't count as "gaps" when they're low
        flags=[f for f in an.get("rda_flags",[]) if f.get("nutrient") not in UPPER_LIMIT_NUTRIENTS]
        n_days=len({f.get("day") for f in flags})
        st.markdown(f"### Nutrient gaps ({len(flags)} below 80% of target, across {n_days} days)")
        if flags:
            fdf=pd.DataFrame(flags).rename(columns={"day":"Day","nutrient":"Nutrient",
                "got":"Actual","target":"Target","pct":"% of target"})
            if "Nutrient" in fdf: fdf["Nutrient"]=fdf["Nutrient"].map(nutrient_label)
            st.dataframe(fdf,width="stretch",hide_index=True)
            st.caption("Note: B12 and vitamin D are commonly supplemented, especially on "
                       "vegan/lactose-free diets - flags here are informational.")
        else: st.success("All tracked nutrients meet >=80% RDA every day.")

AISLES={"Protein":"Protein","Vegetables":"Produce","Fruits":"Produce","Grains":"Grains & bread",
        "Dairy":"Dairy & milks","Fats":"Pantry"}
def grocery_list(plan):
    """Total each food across the week, grouped by store aisle."""
    totals={}
    for d in plan:
        for fs in d["meals"].values():
            for f in fs:
                b=f["_base_food"]
                if b not in totals:
                    totals[b]={"name":short_food_name(f["name"]),"g":0.0,
                               "aisle":AISLES.get(str(f.get("food_group","")),"Other")}
                totals[b]["g"]+=float(f.get("portion_g",0))
    by_aisle={}
    for b,t in totals.items():
        amt=portion_label({"_base_food":b,"portion_g":t["g"]})
        if amt.endswith(" g") and t["g"]>=1000:
            amt=f"{t['g']/1000:.1f} kg"
        by_aisle.setdefault(t["aisle"],[]).append((t["name"],amt,b))
    order=["Produce","Protein","Grains & bread","Dairy & milks","Pantry","Other"]
    return [(a,sorted(by_aisle[a])) for a in order if a in by_aisle]

with tab_groc:
    if not state:
        st.info("Create a plan to get your grocery list.")
    else:
        groups=grocery_list(state["plan"])
        n=sum(len(items) for _,items in groups)
        st.markdown("### Grocery list")
        st.caption(f"{n} items for the whole week, grouped by aisle. Check things off as you shop.")
        cols=st.columns(3)
        for i,(aisle,items) in enumerate(groups):
            with cols[i%3]:
                with st.container(border=True):
                    st.markdown(f"**{aisle}**")
                    for name,amt,b in items:
                        st.checkbox(f"{name} · {amt}",key=f"groc_{b}")

with tab_excl:
    if not state: st.info("Generate a plan to see exclusion explanations.")
    else:
        log=state["log"]
        st.markdown("### What we left out")
        st.caption("This explains what the app removed before building your plan, based on your selections.")
        for stage,label in [("clinical","🩺 Clinical condition"),("allergy","⚠️ Allergen / intolerance"),
                            ("diet","🥬 Diet preference"),("skipped","🔄 Foods you swapped out")]:
            dct=log.get(stage) or {}
            if isinstance(dct,list):  # tolerate list-of-pairs logs too
                dct=dict(dct)
            if stage=="skipped" and not dct:
                continue
            # The database has many brand versions of each food; list each food once.
            edf=pd.DataFrame([{"Food":short_food_name(nm),"Reason":str(reason)}
                              for nm,reason in dct.items()],columns=["Food","Reason"])
            edf=edf.drop_duplicates("Food").sort_values("Food").reset_index(drop=True)
            n=len(edf)
            with st.expander(f"{label} - {n} food{'s' if n!=1 else ''} excluded"):
                if not n:
                    st.write("None.")
                else:
                    st.dataframe(edf,width="stretch",hide_index=True,
                                 height=min(38+35*n,320))

st.markdown("---")
st.caption("NutriAI provides general meal-planning guidance, not medical advice. If you have a "
           "health condition, check your plan with a registered dietitian or physician.")
