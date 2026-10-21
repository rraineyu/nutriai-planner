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
.block-container { padding-top:2.2rem; max-width:1180px; }

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

st.markdown("# 🍽️ NutriAI")
st.markdown("##### Tell us your health needs and NutriAI will create a safe 7-day meal plan with nutrition analysis and explainable exclusions.")

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
        parts.append("removes " + ", ".join(persona.get("allergens", [])) + " allergens")
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

with st.sidebar:
    st.markdown("## Build your plan")
    st.caption("Tell us about yourself and your needs.")

    # Starting values for the form (users always build their own profile)
    d=dict(conditions=[],allergens=[],diet="nonveg",kcal=2000,age=35,sex="female",
           lact=False,pork=False,gf=False,fish=0,na=0,gi=0)

    st.markdown("### 1. About you")
    c1,c2=st.columns(2)
    age=c1.number_input("Age",10,100,d["age"])
    sex=c2.selectbox("Sex",["female","male"],index=0 if d["sex"]=="female" else 1)
    kcal=st.slider("Daily calorie goal",1200,3000,d["kcal"],50)

    st.markdown("### 2. Food preferences")
    diet=st.selectbox("Diet",["nonveg","pescatarian","vegetarian","vegan"],
                      index=["nonveg","pescatarian","vegetarian","vegan"].index(d["diet"]))
    allergens=st.multiselect("Allergies to avoid",ALLERGEN_CHOICES,default=d["allergens"])
    # Two columns instead of three so labels don't get truncated ("Lacto…")
    cc1,cc2=st.columns(2)
    lact=cc1.checkbox("Lactose-free",d["lact"])
    gf=cc2.checkbox("Gluten-free",d["gf"])
    pork=cc1.checkbox("No pork",d["pork"])

    st.markdown("### 3. Health needs")
    conditions=st.multiselect("Conditions",CONDS,default=d["conditions"])

    with st.expander("Advanced settings"):
        st.caption("Optional. Use these if a doctor or dietitian has given you specific limits.")
        na=st.slider("Sodium daily cap (mg, 0=off)",0,2500,d["na"],50)
        gi=st.slider("Max glycemic index (0=off)",0,100,d["gi"],5)
        fish=st.slider("Minimum fish meals/week",0,7,d["fish"])
        plan_style=st.radio("Plan style",["dnn","faiss"],
            format_func=lambda t: TECHNIQUE_LABELS[t],
            help="How NutriAI picks foods for your plan.")
        st.caption(TECHNIQUE_HELP[plan_style])
    technique=plan_style

    go=st.button("Create my 7-day plan",type="primary",width="stretch")

persona=Persona(name="Custom profile",conditions=conditions,
    allergens=allergens,diet=diet,kcal_target=kcal,age=age,sex=sex,lactose_free=lact,
    no_pork=pork,gluten_free=gf,require_fish_meals=fish,sodium_daily_cap=na,gi_max=gi,
    micro_priority=[])

if go:
    with st.spinner("Building your 7-day plan..."):
        t0=time.time()
        plan,log,an=eng.build_plan(persona,technique,seed=time.time_ns()%(1<<30),
                                   avoid=st.session_state.skipped_foods,
                                   prefer=st.session_state.saved_foods)
        gt=time.time()-t0
    if plan is None:
        st.error("No foods match all of your restrictions. Try removing one restriction and create the plan again.")
    else:
        st.session_state.plan={"plan":plan,"log":log.__dict__,"analysis":an,
            "persona":persona.__dict__,"gt":gt,"technique":technique}
        # Streamlit can't auto-close the sidebar, so nudge narrow-screen users
        st.toast("Your 7-day plan is ready! On mobile, close the sidebar (<<) to see it.", icon="✅")

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

tab_plan,tab_an,tab_excl=st.tabs(
    ["My plan 📝 ","Nutrition 📊","Why excluded 🚫"])
state=st.session_state.plan

with tab_plan:
    if not state:
        st.info("**Get started:** fill in your profile in the sidebar, then click "
                "**Create my 7-day plan**.  \n"
                "On a phone or narrow window, tap the **>>** arrow at the top left to open the sidebar.")
    else:
        plan=state["plan"]; an=state["analysis"]
        c1,c2,c3,c4=st.columns(4)
        da=an.get("daily_avg",{})
        kcal_goal=state["persona"].get("kcal_target")
        c1.metric("Avg calories/day",f"{da.get('energy_kcal',0):,.0f}",
                  f"goal {kcal_goal:,}" if kcal_goal else None,delta_color="off")
        c2.metric("Avg protein/day",f"{da.get('protein_g',0):.0f} g")
        c3.metric("Avg fiber/day",f"{da.get('fiber_g',0):.0f} g")
        c4.metric("Fish meals",an.get("fish_meals",0))
        # Small, right-aligned diversity score (kept subtle on purpose)
        div=an.get("diversity_score")
        if div is not None:
            try: div_txt=f"{float(div):.2f}"
            except (TypeError,ValueError): div_txt=str(div)
            st.markdown(
                f'<div style="text-align:right;font-size:0.8rem;color:var(--ink-soft);margin-top:-6px" '
                f'title="How varied your foods are across the week. Higher means less repetition.">'
                f'Diversity score: <b>{div_txt}</b></div>',
                unsafe_allow_html=True)
        st.success(plan_fit_summary(state["persona"], an))
        st.caption("Tap **Swap** to replace any meal you don't like, or **Save** to see those foods more often.")
        st.markdown("---")
        csvb=plan_to_csv(plan); pdfb=plan_to_pdf(plan,state["persona"],an)
        e1,e2=st.columns(2)
        e1.download_button("⬇ Download CSV",csvb,"nutriai_plan.csv","text/csv",width="stretch")
        e2.download_button("⬇ Download PDF",pdfb,"nutriai_plan.pdf","application/pdf",width="stretch")
        st.markdown("---")
        saved=set(st.session_state.saved_foods)
        for day in plan:
            st.markdown(f'<span class="daychip">Day {day["day"]}</span>',unsafe_allow_html=True)
            cols=st.columns(3)
            meal_style={"Breakfast":("🥣","var(--peach)"),
                        "Lunch":("🥗","var(--sage)"),
                        "Dinner":("🍲","var(--lavender)")}
            main_meals=[(m,day["meals"][m]) for m in ("Breakfast","Lunch","Dinner") if m in day["meals"]]
            for col,(meal,foods) in zip(cols,main_meals):
                with col:
                    emoji,accent=meal_style.get(meal,("🍽️","var(--sage)"))
                    title = meal_title(meal, foods)
                    total_kcal = sum(float(f["energy_kcal"])*float(f["_scale"]) for f in foods)
                    total_protein = sum(float(f["protein_g"])*float(f["_scale"]) for f in foods)
                    st.markdown(f"#### {emoji} {meal}")
                    st.markdown(
                        f'<div class="meal-card" style="border-left-color:{accent}">'
                        f'<b style="font-size:1.05rem">{title}</b><br>'
                        f'<small>{ingredient_line(foods)}</small><br>'
                        f'<small>{total_kcal:.0f} kcal · {total_protein:.0f}g protein</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    is_saved = bool(foods) and all(f["_base_food"] in saved for f in foods)
                    b1, b2 = st.columns(2)
                    b1.button("⭐ Saved" if is_saved else "⭐ Save",
                              key=f"save_d{day['day']}_{meal}",
                              help="Use these foods more often in your plans",
                              on_click=on_save, args=(day["day"], meal),
                              disabled=is_saved, width="stretch")
                    b2.button("🔄 Swap",
                              key=f"swap_d{day['day']}_{meal}",
                              help="Replace this meal with a different one",
                              on_click=on_skip, args=(day["day"], meal),
                              width="stretch")
            snack=day["meals"].get("Snack")
            if snack:
                skcal=sum(float(f["energy_kcal"])*float(f["_scale"]) for f in snack)
                st.markdown(
                    f'<div style="margin-top:6px;font-size:0.9rem;color:var(--ink-soft)">'
                    f'🍎 <b>Snack:</b> {ingredient_line(snack)} · {skcal:.0f} kcal</div>',
                    unsafe_allow_html=True)
            st.markdown("")
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

with tab_excl:
    if not state: st.info("Generate a plan to see exclusion explanations.")
    else:
        log=state["log"]
        st.markdown("### Why foods were excluded")
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
