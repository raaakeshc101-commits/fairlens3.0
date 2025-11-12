import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from pathlib import Path
from datetime import datetime

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="FairLens — Performance Review Auditor", layout="wide")
st.title("FairLens — Performance Review Auditor")
st.caption("Detect vague / biased phrases, coach rewrites, and check group fairness on ratings. No PII. n ≥ 5 to aggregate.")

RULES_VERSION = "v1.1-lexicon-60"

# -----------------------------
# Helpers: load rules
# -----------------------------
@st.cache_data
@st.cache_data
def load_bias_rules(path: str = "bias_rules.csv") -> pd.DataFrame:
    from pathlib import Path
    p = Path(path)
    cols = ["phrase","category","context_rule","tip"]
    if not p.exists():
        st.warning("bias_rules.csv not found; using empty rule set.")
        return pd.DataFrame(columns=cols)
    try:
        # Read as pipe-delimited to avoid issues with commas inside tips
        df = pd.read_csv(p, sep="|", engine="python")
    except Exception as e:
        st.error(f"Failed to read bias_rules.csv: {e}")
        return pd.DataFrame(columns=cols)
    # sanity checks
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"bias_rules.csv missing columns: {missing}")
        return pd.DataFrame(columns=cols)
    return df


# -----------------------------
# Smarter flagging utilities
# -----------------------------
POSITIVE_WORDS = ["good","great","improved","improving","excellent","amazing","nice","pleasant","awesome"]
BEHAVIOR_VERBS = ["completed","delivered","reduced","increased","launched","led","designed","documented","resolved","trained","implemented","created","shipped"]

def is_positive_vague(text: str) -> bool:
    t = (text or "").lower()
    has_positive = any(w in t for w in POSITIVE_WORDS)
    has_numbers  = any(ch.isdigit() for ch in t)
    has_behavior = any(v in t for v in BEHAVIOR_VERBS)
    return has_positive and not (has_numbers or has_behavior)

def pattern_match(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, text or "", flags=re.IGNORECASE) is not None
    except re.error:
        return False

def special_context_checks(text: str, gender: str):
    t = (text or "").lower()
    flags = []
    # "fit" only if negated nearby (e.g., "not a good fit")
    for m in re.finditer(r"\bfit\b", t):
        window = t[max(0, m.start()-20):m.start()]
        if "not" in window:
            flags.append(("not a good fit","Bias","Focus on job criteria: name specific gap vs role rubric."))
            break
    # gendered "assertive"
    if ("assertive" in t) and (str(gender).lower() in ["f","female","woman","women","she","her"]):
        flags.append(("assertive (gendered)","Bias","Judge the same behaviors consistently regardless of gender; cite behaviors."))
    return flags

def hybrid_flags(text: str, gender: str, rules_df: pd.DataFrame):
    text_l = (text or "").lower()
    out = []

    # CSV-driven rules
    for _, r in rules_df.iterrows():
        phrase   = str(r["phrase"])
        category = str(r["category"]).strip().title()  # Vague | Bias | Positive-Vague
        rule     = str(r["context_rule"]).strip().lower()
        tip      = str(r.get("tip","")).strip()

        should = False
        if rule == "always":
            should = phrase.lower() in text_l
        elif rule == "pattern":
            should = pattern_match(text, phrase)
        elif rule == "review_context":
            ctx = any(w in text_l for w in ["meeting","feedback","discussion","call","review"])
            should = phrase.lower() in text_l and ctx
        elif rule == "if_gender_female":
            should = phrase.lower() in text_l and str(gender).lower() in ["f","female","woman","women","she","her"]
        else:
            # fallback: substring
            should = phrase.lower() in text_l

        if should:
            out.append({"phrase": phrase, "category": category, "tip": tip})

    # Heuristic: positive praise without evidence
    if is_positive_vague(text):
        out.append({"phrase":"positive-without-evidence","category":"Vague",
                    "tip":"Add a number or behavior (who/what/result). e.g., “Completed 95% of Q3 deliverables.”"})

    # Special context checks
    for ph, cat, tip in special_context_checks(text, gender):
        out.append({"phrase": ph, "category": cat, "tip": tip})

    return out

# -----------------------------
# Seed session reviews (anonymized)
# -----------------------------
if "reviews" not in st.session_state:
    st.session_state.reviews = pd.DataFrame({
        "employee_id": [f"E{i:03d}" for i in range(1, 11)],
        "role": ["Manager"]*5 + ["Analyst"]*5,
        "gender": ["F","M","F","M","F","M","F","M","F","M"],
        "kpi_rating": [4,3,4,3,4,3,3,3,4,3],
        "competency_rating": [4,4,3,3,4,3,3,3,4,3],
        "initiative_rating": [4,3,3,3,4,3,3,3,4,3],
        "overall_rating": [4,3,3,3,4,3,3,3,4,3],
        "comment": [
            "Strong potential; team player.",
            "Good attitude; average execution.",
            "Works well under pressure; sometimes too energetic.",
            "Not a good cultural fit. Hard worker though.",
            "Great attitude; on-time delivery.",
            "Can be emotional in feedback.",
            "Average performance; could do better.",
            "Aggressive in meetings.",
            "Great culture fit.",
            "Bossy in team settings."
        ]
    })

# -----------------------------
# Tabs
# -----------------------------
tab_submit, tab_audit, tab_priv = st.tabs(["Submit Review", "Audit & Fairness", "Privacy & Export"])

# -----------------------------
# TAB 1: Submit
# -----------------------------
with tab_submit:
    st.subheader("Submit a Performance Review (Anonymized)")
    st.write("Use anonymized IDs only. Provide role & gender for group fairness. Avoid names/PII.")

    with st.form("review_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            employee_id = st.text_input("Employee ID (e.g., E011)")
        with c2:
            role = st.selectbox("Role", ["Manager", "Analyst", "Engineer", "Sales", "Other"])
        with c3:
            gender = st.selectbox("Gender (for parity demo)", ["F","M","Non-binary/Other"])

        st.markdown("**Ratings (1–5)**")
        r1, r2, r3, r4 = st.columns(4)
        with r1: kpi = st.slider("KPI", 1, 5, 3)
        with r2: comp = st.slider("Competency", 1, 5, 3)
        with r3: initv = st.slider("Initiative", 1, 5, 3)
        with r4: overall = st.slider("Overall", 1, 5, 3)

        comment = st.text_area("Manager Comment (no PII)",
                               value="You have come a long way this year but you need to gain more product knowledge.",
                               height=140)
        submitted = st.form_submit_button("Save Review", type="primary")

    if submitted:
        if not employee_id.strip():
            st.error("Please provide an anonymized Employee ID.")
        else:
            new_row = pd.DataFrame([{
                "employee_id": employee_id.strip(),
                "role": role,
                "gender": gender,
                "kpi_rating": kpi,
                "competency_rating": comp,
                "initiative_rating": initv,
                "overall_rating": overall,
                "comment": comment.strip()
            }])
            st.session_state.reviews = pd.concat([st.session_state.reviews, new_row], ignore_index=True)
            st.success(f"Saved review for {employee_id.strip()} ✅")

    st.markdown("#### Current (Anonymized) Reviews")
    st.dataframe(st.session_state.reviews, use_container_width=True, height=320)

# -----------------------------
# TAB 2: Audit & Fairness
# -----------------------------
with tab_audit:
    st.subheader("Audit & Fairness")

    uploaded = st.file_uploader("Optional: Upload CSV to replace session data (columns must match).", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            needed = {"employee_id","role","gender","kpi_rating","competency_rating","initiative_rating","overall_rating","comment"}
            if not needed.issubset(df_up.columns):
                st.error(f"CSV missing columns. Required: {sorted(list(needed))}")
            else:
                st.session_state.reviews = df_up.copy()
                st.success("Uploaded CSV loaded into session.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    df = st.session_state.reviews.copy()

    # ---- Flags
    st.markdown("### Narrative Flags (Vague / Bias) with Coaching Tips")
    rules_df = load_bias_rules()
    all_flags = []
    for _, row in df.iterrows():
        flags = hybrid_flags(row.get("comment",""), row.get("gender",""), rules_df)
        for f in flags:
            all_flags.append({
                "employee_id": row.get("employee_id",""),
                "role": row.get("role",""),
                "gender": row.get("gender",""),
                "phrase": f["phrase"],
                "category": f["category"],
                "tip": f.get("tip","")
            })

    if all_flags:
        flag_df = pd.DataFrame(all_flags)
        st.dataframe(flag_df, use_container_width=True, height=260)
        by_cat = flag_df.groupby("category")["phrase"].count().rename("count").reset_index()
        st.bar_chart(by_cat.set_index("category"))
        st.caption("Use flags as coaching signals; rewrite with behavior-based evidence.")
    else:
        st.info("No flags detected by current rules.")

    st.markdown("---")

    # ---- Group fairness
    st.markdown("### Ratings Fairness (Mean Gap + AIR)")
    by_group = st.selectbox("Compare by group", ["gender","role"])
    cols = ["kpi_rating","competency_rating","initiative_rating","overall_rating"]
    summary = df.groupby(by_group)[cols].agg(["mean","count"]).round(2)
    st.dataframe(summary, use_container_width=True)

    # Mean gap (overall)
    groups = df[by_group].dropna().unique()
    if len(groups) >= 2:
        counts = df[by_group].value_counts().index.tolist()
        g1 = counts[0]
        g2 = counts[1] if len(counts) > 1 else groups[1]
        m1 = df[df[by_group]==g1]["overall_rating"].mean()
        m2 = df[df[by_group]==g2]["overall_rating"].mean()
        gap = abs(m1 - m2)
        st.write(f"{by_group}={g1}: {m1:.2f}  vs  {by_group}={g2}: {m2:.2f}  → **Gap = {gap:.2f}**")
        if gap >= 0.30:
            st.warning("Gap ≥ 0.30 (1–5 scale) — consider calibration review.")
        else:
            st.success("Mean gap < 0.30 — no strong disparity indicated.")
    else:
        st.info("Provide at least two groups to compute a gap.")

    # AIR proxy
    st.markdown("#### Meets/Exceeds Parity (AIR)")
    threshold = st.slider("Meets/Exceeds threshold (Overall ≥)", 1.0, 5.0, 3.0, 0.5)
    rates = (df.assign(meets=(df["overall_rating"] >= threshold))
               .groupby(by_group)["meets"].mean()
               .rename("rate")
               .reset_index())
    st.dataframe(rates, use_container_width=True)
    if len(rates) >= 2:
        top = rates["rate"].max()
        bottom = rates["rate"].min()
        air = (bottom / top) if top > 0 else np.nan
        st.write(f"AIR (min/max) = **{air:.2f}** (rule-of-thumb ≥ 0.80)")
        if air < 0.80:
            st.error("AIR < 0.80 — investigate disparity (sample size, criteria clarity, rater training).")
        else:
            st.success("AIR ≥ 0.80 — no adverse impact signal on this proxy.")
    else:
        st.info("Provide at least two groups to compute AIR.")

# -----------------------------
# TAB 3: Privacy & Export
# -----------------------------
with tab_priv:
    st.subheader("Privacy, Governance & Export")
    st.markdown(f"""
- **No PII**: Use anonymized IDs only.
- **Aggregation-first**: Group metrics only when **n ≥ 5** per group.
- **Retention**: Session-based; no server persistence. Export locally if needed.
- **Explainability**: Rule-based flags are transparent and editable via **bias_rules.csv**.
- **Compliance touchpoints**: Title VII principles; AIR (4/5ths) as a rule-of-thumb.
- **Rules version**: `{RULES_VERSION}`
    """)

    # Export current session data
    buf = io.StringIO()
    st.session_state.reviews.to_csv(buf, index=False)
    st.download_button("Download Current Reviews CSV", buf.getvalue(), file_name="fairlens_reviews.csv")

    # Export flags for audit (recompute once)
    rules_df = load_bias_rules()
    flags = []
    for _, row in st.session_state.reviews.iterrows():
        f = hybrid_flags(row.get("comment",""), row.get("gender",""), rules_df)
        for x in f:
            flags.append({
                "employee_id": row.get("employee_id",""),
                "role": row.get("role",""),
                "gender": row.get("gender",""),
                "phrase": x["phrase"],
                "category": x["category"],
                "tip": x.get("tip","")
            })
    if flags:
        buf2 = io.StringIO()
        pd.DataFrame(flags).to_csv(buf2, index=False)
        st.download_button("Download Flags CSV (Audit Snapshot)", buf2.getvalue(), file_name=f"flags_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

st.caption("© FairLens — educational demo. Replace lists/thresholds with your org standards before deployment.")
