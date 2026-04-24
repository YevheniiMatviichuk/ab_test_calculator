import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson_means_test, gamma, chisquare
from scipy.stats.mstats import winsorize
from statsmodels.stats.proportion import (
    proportions_ztest,
    confint_proportions_2indep,
    proportion_effectsize,
)
from statsmodels.stats.power import NormalIndPower
import math

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test Calculator",
    page_icon="🧪",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }
    .result-box {
        background: #f0f4ff;
        border-left: 4px solid #0066cc;
        padding: 20px 24px;
        border-radius: 0 8px 8px 0;
        margin-top: 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        line-height: 2;
        color: #111111 !important;
    }
    .result-box-win  { border-left: 4px solid #00875a; background: #e6f9f0; color: #111111 !important; }
    .result-box-loss { border-left: 4px solid #cc0000; background: #fff0f0; color: #111111 !important; }
    .result-box-warn { border-left: 4px solid #cc7700; background: #fff8e6; color: #111111 !important; }
    .result-box strong { color: #111111 !important; }
    .param-box {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        font-size: 13px;
        line-height: 1.8;
        color: #333;
    }
    .param-box b { font-family: 'IBM Plex Mono', monospace; color: #0066cc; }
    .note { font-size: 12px; color: #888; font-style: italic; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 A/B Calculator")
    st.markdown("---")
    page = st.radio(
        "Select calculator",
        options=[
            "⭐ mSPRT — Any Conversion (Default)",
            "Poisson — Rare Conversions (<0.5%)",
            "Z-Test — Normal Conversions (≥0.5%)",
            "Bootstrap — Average per User",
            "Sample Size & Power",
            "SRM Check — Traffic Split Validity",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="font-size:11px; color:#666; font-family: monospace;">'
        'Built by Data team<br>Based on internal A/B framework</p>',
        unsafe_allow_html=True,
    )

# ── Shared param guide ────────────────────────────────────────────────────────
def show_param_guide(show_mde=True, show_alpha=True, show_bootstrap=False,
                     show_winsorize=False, show_mde_note=None):
    lines = []
    if show_mde:
        mde_note = show_mde_note or (
            "The smallest relative lift you and the PM agreed is worth shipping. "
            "Must be defined before launch, not after seeing data. "
            "E.g. 10 means 'we only ship if lift is at least +10%'."
        )
        lines.append(f"<b>MDE (%)</b> — Minimum Detectable Effect: {mde_note}")
    if show_alpha:
        lines.append(
            "<b>Significance level (α)</b> — The false positive risk you accept. "
            "Standard is 0.05 (5%). Use 0.01 for higher-stakes decisions. "
            "Use 0.025 if testing two primary metrics (Bonferroni correction)."
        )
    if show_bootstrap:
        lines.append(
            "<b>Bootstrap iterations</b> — How many times to resample the data. "
            "More = more precise p-value, but slower. 10,000 is standard."
        )
    if show_winsorize:
        lines.append(
            "<b>Winsorize top %</b> — Caps extreme outlier values at this percentile. "
            "Default 1% = 99th percentile cap. Reduces outlier distortion without removing data."
        )
    st.markdown(
        '<div class="param-box">' + "<br>".join(lines) + '</div>',
        unsafe_allow_html=True
    )

# ── Shared alpha input ────────────────────────────────────────────────────────
def alpha_input(key, default=0.05):
    return st.number_input(
        "Significance level (α)",
        min_value=0.001, max_value=0.20,
        value=default, step=0.005, format="%.3f",
        key=key,
        help="Standard: 0.05. Use 0.025 for two primary metrics. Use 0.01 for SRM or high-stakes tests."
    )

# ── Shared verdict box ────────────────────────────────────────────────────────
def show_result(rate_c, rate_t, lift, lift_ci, p_value, p_label, alpha, mde, ci_positive):
    significant = p_value < alpha
    above_mde   = lift >= mde
    ship        = significant and above_mde and ci_positive
    box_class   = "result-box-win" if ship else "result-box-loss"
    verdict     = "✅ SHIP" if ship else "❌ DO NOT SHIP"

    st.markdown(f"""
    <div class="result-box {box_class}">
        <strong>Verdict: {verdict}</strong><br><br>
        Rate control &nbsp;&nbsp;&nbsp;&nbsp;: {rate_c:.4%}<br>
        Rate treatment &nbsp;: {rate_t:.4%}<br>
        Relative lift &nbsp;&nbsp;: {lift:+.2%} &nbsp;(95% CI: {lift_ci[0]:+.2%} to {lift_ci[1]:+.2%})<br>
        {p_label}: {p_value:.4f}<br>
        Significant &nbsp;&nbsp;&nbsp;&nbsp;: {"Yes" if significant else "No"} (alpha={alpha})<br>
        Lift >= MDE &nbsp;&nbsp;&nbsp;&nbsp;: {"Yes" if above_mde else "No"} (MDE={mde:.1%})<br>
        CI lower > 0 &nbsp;&nbsp;: {"Yes" if ci_positive else "No"}
    </div>
    """, unsafe_allow_html=True)

    if not significant and above_mde:
        st.warning("Lift looks promising but not yet significant. Continue running.")
    elif significant and not above_mde:
        st.error("Significant but below MDE — effect too small to matter. Do not ship.")
    elif not significant and not above_mde and lift_ci[1] < mde:
        st.info("Confident null — CI rules out meaningful effects. Move on.")


# ══════════════════════════════════════════════════════════════════════════════
# mSPRT — DEFAULT
# ══════════════════════════════════════════════════════════════════════════════
if page == "⭐ mSPRT — Any Conversion (Default)":
    st.markdown("# mSPRT — Sequential Testing")
    st.markdown(
        "**Our default calculator for conversion metrics.** "
        "Produces an always-valid p-value — safe to check at any point "
        "during the experiment without inflating your false positive rate. "
        "Use this whenever results are checked more than once, which in practice is always."
    )
    st.info("Replaces Poisson and Z-Test calculators when results are checked mid-experiment. Works for both rare and normal conversion rates.")
    st.markdown("---")

    show_param_guide(
        show_mde=True, show_alpha=True,
        show_mde_note=(
            "The smallest relative lift you and the PM agreed is worth shipping. "
            "This also tunes the test sensitivity — set it to the same value agreed before launch. "
            "E.g. 10 means 'we only ship if lift is at least +10%'."
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Treatment**")
        conv_t  = st.number_input("Conversions in treatment", min_value=0, value=12, key="t_conv4")
        users_t = st.number_input("Total users in treatment", min_value=1, value=100, key="t_users4")
    with col2:
        st.markdown("**Control**")
        conv_c  = st.number_input("Conversions in control",   min_value=0, value=10, key="c_conv4")
        users_c = st.number_input("Total users in control",   min_value=1, value=100, key="c_users4")

    st.markdown("**Parameters**")
    col3, col4 = st.columns(2)
    with col3:
        mde = st.number_input(
            "MDE — minimum relative lift worth shipping (%)",
            min_value=0.1, max_value=100.0, value=10.0, step=0.5, key="mde4"
        ) / 100
    with col4:
        alpha = alpha_input("alpha4")

    if st.button("Calculate", key="btn4", type="primary"):
        rate_t     = conv_t / users_t
        rate_c     = conv_c / users_c
        diff       = rate_t - rate_c
        n          = (users_t + users_c) / 2
        p_pool     = (conv_t + conv_c) / (users_t + users_c)
        sigma_sq_D = 2 * p_pool * (1 - p_pool)
        tau_sq     = (rate_c * mde) ** 2
        denom      = n * tau_sq + sigma_sq_D
        Lambda     = np.sqrt(sigma_sq_D / denom) * np.exp(
                         (n**2 * tau_sq * diff**2) / (2 * sigma_sq_D * denom))
        p_value    = float(min(1.0, 1 / Lambda))
        lift       = diff / rate_c
        ci_abs     = confint_proportions_2indep(conv_t, users_t, conv_c, users_c)
        lift_ci    = (ci_abs[0] / rate_c, ci_abs[1] / rate_c)

        show_result(rate_c, rate_t, lift, lift_ci, p_value,
                    "Always-valid p", alpha, mde, ci_abs[0] > 0)


# ══════════════════════════════════════════════════════════════════════════════
# POISSON — RARE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Poisson — Rare Conversions (<0.5%)":
    st.markdown("# Poisson Test — Rare Conversions (<0.5%)")
    st.markdown(
        "Use when conversion rate is below ~0.5% in either variant. "
        "At very low rates, the normal approximation breaks down — "
        "Poisson test is built specifically for rare event counts and handles this correctly. "
        "Returns a fixed-horizon p-value: valid only if results are read once at the end. "
        "If you check results mid-experiment, use mSPRT instead."
    )
    st.markdown("---")

    show_param_guide(show_mde=True, show_alpha=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Treatment**")
        conv_t  = st.number_input("Conversions in treatment", min_value=0, value=3,   key="t_conv1")
        users_t = st.number_input("Total users in treatment", min_value=1, value=1000, key="t_users1")
    with col2:
        st.markdown("**Control**")
        conv_c  = st.number_input("Conversions in control",   min_value=0, value=2,   key="c_conv1")
        users_c = st.number_input("Total users in control",   min_value=1, value=1000, key="c_users1")

    st.markdown("**Parameters**")
    col3, col4 = st.columns(2)
    with col3:
        mde = st.number_input(
            "MDE — minimum relative lift worth shipping (%)",
            min_value=0.1, max_value=100.0, value=10.0, step=0.5, key="mde1"
        ) / 100
    with col4:
        alpha = alpha_input("alpha1")

    if st.button("Calculate", key="btn1", type="primary"):
        rate_t  = conv_t / users_t
        rate_c  = conv_c / users_c
        result  = poisson_means_test(conv_t, users_t, conv_c, users_c)
        p_value = result.pvalue

        def poisson_ci(k, n, a):
            return (gamma.ppf(a / 2, k) / n, gamma.ppf(1 - a / 2, k + 1) / n)

        ci_t    = poisson_ci(conv_t, users_t, alpha)
        ci_c    = poisson_ci(conv_c, users_c, alpha)
        lift    = (rate_t - rate_c) / rate_c
        lift_ci = ((ci_t[0] - rate_c) / rate_c, (ci_t[1] - rate_c) / rate_c)

        significant = p_value < alpha
        above_mde   = lift >= mde
        ci_positive = ci_t[0] > rate_c
        ship        = significant and above_mde and ci_positive
        box_class   = "result-box-win" if ship else "result-box-loss"
        verdict     = "✅ SHIP" if ship else "❌ DO NOT SHIP"

        st.markdown(f"""
        <div class="result-box {box_class}">
            <strong>Verdict: {verdict}</strong><br><br>
            Rate treatment : {rate_t:.4%} &nbsp;(95% CI: {ci_t[0]:.4%} – {ci_t[1]:.4%})<br>
            Rate control &nbsp;&nbsp;: {rate_c:.4%} &nbsp;(95% CI: {ci_c[0]:.4%} – {ci_c[1]:.4%})<br>
            Relative lift &nbsp;: {lift:+.2%} &nbsp;(95% CI: {lift_ci[0]:+.2%} – {lift_ci[1]:+.2%})<br>
            P-value &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {p_value:.4f}<br>
            Significant &nbsp;&nbsp;: {"Yes" if significant else "No"} (alpha={alpha})<br>
            Lift >= MDE &nbsp;&nbsp;: {"Yes" if above_mde else "No"} (MDE={mde:.1%})<br>
            CI lower > 0 : {"Yes" if ci_positive else "No"}
        </div>
        """, unsafe_allow_html=True)

        if not significant and above_mde:
            st.warning("Lift looks promising but not yet significant. Continue running.")
        elif significant and not above_mde:
            st.error("Significant but below MDE — effect too small to matter. Do not ship.")


# ══════════════════════════════════════════════════════════════════════════════
# Z-TEST — NORMAL CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Z-Test — Normal Conversions (≥0.5%)":
    st.markdown("# Z-Test — Normal Conversions (≥0.5%)")
    st.markdown(
        "Use when conversion rate is 0.5% or above in both variants. "
        "At this level the binomial is well approximated by the normal distribution. "
        "Returns a fixed-horizon p-value: valid only if results are read once at the end. "
        "If you check results mid-experiment, use mSPRT instead."
    )
    st.markdown("---")

    show_param_guide(show_mde=True, show_alpha=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Treatment**")
        conv_t  = st.number_input("Conversions in treatment", min_value=0, value=12,  key="t_conv2")
        users_t = st.number_input("Total users in treatment", min_value=1, value=100, key="t_users2")
    with col2:
        st.markdown("**Control**")
        conv_c  = st.number_input("Conversions in control",   min_value=0, value=10,  key="c_conv2")
        users_c = st.number_input("Total users in control",   min_value=1, value=100, key="c_users2")

    st.markdown("**Parameters**")
    col3, col4 = st.columns(2)
    with col3:
        mde = st.number_input(
            "MDE — minimum relative lift worth shipping (%)",
            min_value=0.1, max_value=100.0, value=10.0, step=0.5, key="mde2"
        ) / 100
    with col4:
        alpha = alpha_input("alpha2")

    if st.button("Calculate", key="btn2", type="primary"):
        rate_t     = conv_t / users_t
        rate_c     = conv_c / users_c
        _, p_value = proportions_ztest(
            np.array([conv_t, conv_c]),
            np.array([users_t, users_c])
        )
        ci_abs  = confint_proportions_2indep(conv_t, users_t, conv_c, users_c, alpha=alpha)
        lift    = (rate_t - rate_c) / rate_c
        lift_ci = (ci_abs[0] / rate_c, ci_abs[1] / rate_c)

        show_result(rate_c, rate_t, lift, lift_ci, p_value,
                    "P-value", alpha, mde, ci_abs[0] > 0)


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP — CONTINUOUS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Bootstrap — Average per User":
    st.markdown("# Bootstrap — Average per User")
    st.markdown(
        "Use for metrics like views, time spent, or revenue per user — anything that is "
        "an average rather than a conversion rate. "
        "Winsorizes (caps extreme outliers) then uses bootstrap for significance testing. "
        "Returns a fixed-horizon p-value: valid only if results are read once at the end."
    )
    st.markdown("---")

    show_param_guide(
        show_mde=True, show_alpha=True,
        show_bootstrap=True, show_winsorize=True
    )

    # ── CSV upload ────────────────────────────────────────────────────────────
    st.markdown("**Upload data**")
    st.markdown(
        '<p class="note">Upload a CSV with two columns: one for per-user metric values in treatment, '
        'one for control. Column names do not matter — first column = treatment, second = control. '
        'One row per user.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    t_data = None
    c_data = None

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] < 2:
                st.error("CSV must have at least 2 columns — treatment and control values.")
            else:
                col_names = df.columns.tolist()
                st.success(f"Loaded {len(df):,} rows. Columns detected: {col_names[0]} (treatment), {col_names[1]} (control)")

                t_col = st.selectbox("Select treatment column", options=col_names, index=0)
                c_col = st.selectbox("Select control column",   options=col_names, index=1)

                t_data = df[t_col].dropna().values.astype(float)
                c_data = df[c_col].dropna().values.astype(float)

                col_prev1, col_prev2 = st.columns(2)
                with col_prev1:
                    st.markdown(f"**Treatment** — {len(t_data):,} users, mean = {t_data.mean():.4f}")
                with col_prev2:
                    st.markdown(f"**Control** — {len(c_data):,} users, mean = {c_data.mean():.4f}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.markdown('<p class="note">No file uploaded — using example data below for preview.</p>', unsafe_allow_html=True)
        t_data = np.array([3,1,0,5,2,8,0,1,4,2,0,0,3,1,6], dtype=float)
        c_data = np.array([2,0,1,3,1,4,0,0,2,1,0,1,2,0,3], dtype=float)

    st.markdown("**Parameters**")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        mde = st.number_input(
            "MDE (%)", min_value=0.1, max_value=100.0,
            value=10.0, step=0.5, key="mde3",
            help="Minimum relative lift worth shipping."
        ) / 100
    with col4:
        alpha = alpha_input("alpha3")
    with col5:
        n_bootstrap = st.number_input(
            "Bootstrap iterations", min_value=1000, max_value=100000,
            value=10000, step=1000, key="n_boot",
            help="More = more precise but slower. 10,000 is standard."
        )
    with col6:
        wins_pct = st.number_input(
            "Winsorize top %", min_value=0.1, max_value=10.0,
            value=1.0, step=0.1, key="wins",
            help="Caps values above this percentile. Default 1% = 99th pct cap."
        ) / 100

    if st.button("Calculate", key="btn3", type="primary"):
        if t_data is None or c_data is None:
            st.error("No data loaded. Please upload a CSV file.")
        else:
            try:
                with st.spinner("Running bootstrap... this may take a few seconds."):
                    t_clean  = np.array(winsorize(t_data, limits=[0, wins_pct]))
                    c_clean  = np.array(winsorize(c_data, limits=[0, wins_pct]))
                    obs_diff = t_clean.mean() - c_clean.mean()

                    diffs = np.array([
                        np.random.choice(t_clean, len(t_clean), replace=True).mean() -
                        np.random.choice(c_clean, len(c_clean), replace=True).mean()
                        for _ in range(int(n_bootstrap))
                    ])
                    ci = np.percentile(diffs, [alpha / 2 * 100, (1 - alpha / 2) * 100])

                    t_shifted  = t_clean - t_clean.mean() + c_clean.mean()
                    null_diffs = np.array([
                        np.random.choice(t_shifted, len(t_shifted), replace=True).mean() -
                        np.random.choice(c_clean,   len(c_clean),   replace=True).mean()
                        for _ in range(int(n_bootstrap))
                    ])
                    p_value = float(np.mean(np.abs(null_diffs) >= np.abs(obs_diff)))
                    lift    = obs_diff / c_clean.mean()
                    lift_ci = (ci[0] / c_clean.mean(), ci[1] / c_clean.mean())

                significant = p_value < alpha
                above_mde   = lift >= mde
                ci_positive = ci[0] > 0
                ship        = significant and above_mde and ci_positive
                box_class   = "result-box-win" if ship else "result-box-loss"
                verdict     = "✅ SHIP" if ship else "❌ DO NOT SHIP"

                st.markdown(f"""
                <div class="result-box {box_class}">
                    <strong>Verdict: {verdict}</strong><br><br>
                    Mean control &nbsp;&nbsp;&nbsp;&nbsp;: {c_clean.mean():.4f}<br>
                    Mean treatment &nbsp;: {t_clean.mean():.4f}<br>
                    Absolute diff &nbsp;&nbsp;: {obs_diff:+.4f} &nbsp;(95% CI: {ci[0]:+.4f} to {ci[1]:+.4f})<br>
                    Relative lift &nbsp;&nbsp;: {lift:+.2%} &nbsp;(95% CI: {lift_ci[0]:+.2%} to {lift_ci[1]:+.2%})<br>
                    P-value &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {p_value:.4f}<br>
                    Significant &nbsp;&nbsp;&nbsp;&nbsp;: {"Yes" if significant else "No"} (alpha={alpha})<br>
                    Lift >= MDE &nbsp;&nbsp;&nbsp;&nbsp;: {"Yes" if above_mde else "No"} (MDE={mde:.1%})<br>
                    CI lower > 0 &nbsp;&nbsp;: {"Yes" if ci_positive else "No"}
                </div>
                """, unsafe_allow_html=True)

                if not significant and above_mde:
                    st.warning("Lift looks promising but not yet significant. Continue running.")
                elif significant and not above_mde:
                    st.error("Significant but below MDE — effect too small to matter. Do not ship.")
                elif not significant and not above_mde and lift_ci[1] < mde:
                    st.info("Confident null — CI rules out meaningful effects. Move on.")

            except Exception as e:
                st.error(f"Calculation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE SIZE & POWER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sample Size & Power":
    st.markdown("# Sample Size & Power")
    st.markdown(
        "Calculate how many users you need per variant to reliably detect your MDE. "
        "Run this **before** launching any test."
    )
    st.markdown("---")

    st.markdown(
        '<div class="param-box">'
        "<b>Baseline conversion rate (%)</b> — Your current conversion rate before the test. "
        "Pull this from Mixpanel over the last 4 weeks.<br>"
        "<b>MDE (%)</b> — Minimum relative lift worth shipping. Agree with PM before launch. "
        "Smaller MDE = more users needed.<br>"
        "<b>Statistical power</b> — Probability of detecting a real effect if one exists. "
        "Standard is 80%. Higher power = more users needed.<br>"
        "<b>Significance level (α)</b> — False positive risk. Standard is 0.05."
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        baseline = st.number_input(
            "Baseline conversion rate (%)",
            min_value=0.001, max_value=100.0, value=10.0, step=0.1,
            help="Current conversion rate — pull from Mixpanel, last 4 weeks."
        ) / 100
        mde_ss = st.number_input(
            "MDE — minimum relative lift worth shipping (%)",
            min_value=0.1, max_value=100.0, value=10.0, step=0.5,
            help="Smaller MDE = more users needed."
        ) / 100
    with col2:
        power_ss = st.selectbox(
            "Statistical power",
            options=[0.70, 0.80, 0.90], index=1,
            help="80% is the industry standard. 90% for higher-stakes tests."
        )
        alpha_ss = alpha_input("alpha_ss")

    if st.button("Calculate sample size", type="primary"):
        target_rate = baseline * (1 + mde_ss)
        effect_size = proportion_effectsize(target_rate, baseline)
        analysis    = NormalIndPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha_ss,
            power=power_ss,
            alternative='two-sided'
        )
        n_per = math.ceil(sample_size)

        st.markdown(f"""
        <div class="result-box" style="color:#111111 !important;">
            Baseline rate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {baseline:.2%}<br>
            Target rate (MDE) &nbsp;&nbsp;: {target_rate:.2%} &nbsp;(+{mde_ss:.1%} relative)<br>
            Statistical power &nbsp;&nbsp;: {power_ss:.0%}<br>
            Significance (alpha) : {alpha_ss}<br><br>
            <strong>Required per variant &nbsp;: {n_per:,} users</strong><br>
            <strong>Total required &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {n_per * 2:,} users</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<p class="note">Minimum test duration: 2 full weeks regardless of when sample size is reached.</p>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SRM CHECK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SRM Check — Traffic Split Validity":
    st.markdown("# SRM Check — Traffic Split Validity")
    st.markdown(
        "**Run this before looking at any metric results.** "
        "SRM (Sample Ratio Mismatch) checks whether your variants received the traffic split you intended. "
        "If the split is off, your groups are not comparable and any results are unreliable — "
        "regardless of p-value or lift."
    )
    st.info("Uses α=0.01 by default — stricter than standard tests because this is a validity check, not a business decision.")
    st.markdown("---")

    st.markdown(
        '<div class="param-box">'
        "<b>Users per variant</b> — Number of users assigned to each variant.<br>"
        "<b>Expected split</b> — The intended traffic proportion per variant. "
        "Leave all at 0 for equal split. For 70/30 enter 0.7 and 0.3.<br>"
        "<b>Significance level (α)</b> — Default 0.01. SRM is a validity gate, not a business decision.<br>"
        "<b>Practical deviation threshold (%)</b> — At large scale (millions of users), "
        "even random rounding in assignment logic is statistically significant. "
        "This threshold requires the imbalance to also be practically meaningful before flagging SRM. "
        "SRM is only flagged when BOTH the p-value and this threshold are breached. "
        "Default 1% is recommended for high-traffic experiments."
        '</div>',
        unsafe_allow_html=True
    )

    n_variants = st.number_input(
        "Number of variants (including control)",
        min_value=2, max_value=6, value=2, step=1
    )

    st.markdown("**Observed user counts**")
    obs_cols = st.columns(int(n_variants))
    observed_counts = []
    variant_labels  = ["Control"] + [f"Treatment {i}" for i in range(1, int(n_variants))]
    for i, col in enumerate(obs_cols):
        with col:
            val = col.number_input(
                variant_labels[i],
                min_value=0, value=1000,
                key=f"obs_{i}"
            )
            observed_counts.append(val)

    st.markdown("**Expected split** (leave all at 0 for equal split)")
    exp_cols = st.columns(int(n_variants))
    expected_split_raw = []
    for i, col in enumerate(exp_cols):
        with col:
            val = col.number_input(
                f"{variant_labels[i]} proportion",
                min_value=0.0, max_value=1.0,
                value=0.0, step=0.05, format="%.2f",
                key=f"exp_{i}"
            )
            expected_split_raw.append(val)

    # SRM uses 0.01 as default
    alpha_srm = alpha_input("alpha_srm", default=0.01)

    practical_threshold = st.number_input(
        "Practical deviation threshold (%)",
        min_value=0.1, max_value=20.0, value=1.0, step=0.1, format="%.1f",
        help=(
            "At large scale (millions of users), even tiny random imbalances "
            "are statistically significant. This threshold adds a practical gate: "
            "SRM is only flagged if any variant also deviates from its expected "
            "share by more than this % in relative terms. "
            "Default 1% means a planned 50% variant must be outside 49.5%–50.5% to flag. "
            "Recommended: 1–2% for large-scale experiments."
        )
    ) / 100

    if st.button("Run SRM Check", type="primary"):
        observed = np.array(observed_counts, dtype=float)

        # ── Zero check ────────────────────────────────────────────────────────
        if observed.sum() == 0:
            st.error("All variant counts are zero. Please enter observed user counts.")
            st.stop()
        if any(o == 0 for o in observed):
            st.warning(
                "One or more variants have zero users. This will almost certainly "
                "be flagged as SRM. Verify your assignment logging before proceeding."
            )

        total = observed.sum()

        expected_split = (
            [1 / len(observed)] * len(observed)
            if sum(expected_split_raw) == 0
            else expected_split_raw
        )

        expected_counts = np.array(expected_split) * total
        stat, p_value   = chisquare(f_obs=observed, f_exp=expected_counts)

        # ── Practical deviation check ─────────────────────────────────────────
        # Relative deviation = |observed_share - expected_share| / expected_share
        obs_shares = observed / total
        exp_shares = np.array(expected_split)
        rel_devs   = np.abs(obs_shares - exp_shares) / exp_shares
        max_rel_dev        = rel_devs.max()
        max_rel_dev_idx    = rel_devs.argmax()
        practical_breach   = max_rel_dev > practical_threshold

        # SRM only if BOTH statistical and practical thresholds are breached
        srm_detected = p_value < alpha_srm and practical_breach

        obs_pcts   = [f"{o/total:.3%}" for o in observed]
        exp_pcts   = [f"{e:.3%}" for e in expected_split]
        exp_counts_fmt = [f"{e:,.0f}" for e in expected_counts]

        if srm_detected:
            box_class = "result-box-warn"
            verdict   = "⚠️ SRM DETECTED — do not analyze results"
        elif p_value < alpha_srm and not practical_breach:
            box_class = "result-box-win"
            verdict   = "✅ No practical SRM — statistically significant but deviation is within acceptable range"
        else:
            box_class = "result-box-win"
            verdict   = "✅ No SRM — split looks healthy"

        rows = ""
        for i in range(len(observed)):
            dev_flag = " ⚠️" if i == max_rel_dev_idx and practical_breach else ""
            rows += (
                f"{variant_labels[i]} &nbsp;&nbsp;: "
                f"{observed[i]:,.0f} observed ({obs_pcts[i]}) "
                f"vs {exp_counts_fmt[i]} expected ({exp_pcts[i]})"
                f"{dev_flag}<br>"
            )

        st.markdown(f"""
        <div class="result-box {box_class}">
            <strong>{verdict}</strong><br><br>
            {rows}<br>
            Chi-squared stat &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {stat:.4f}<br>
            P-value &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {p_value:.6f}<br>
            Statistical threshold &nbsp;: {alpha_srm}<br>
            Max relative deviation : {max_rel_dev:.3%} ({variant_labels[max_rel_dev_idx]})<br>
            Practical threshold &nbsp;&nbsp;: {practical_threshold:.1%}<br>
            Practical breach &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {"Yes ⚠️" if practical_breach else "No ✅"}
        </div>
        """, unsafe_allow_html=True)

        if srm_detected:
            st.error(
                "SRM detected — both statistical and practical thresholds breached. "
                "Do not proceed with metric analysis. "
                "Investigate: assignment logic bugs, duplicate exposures, "
                "bot traffic, or logging inconsistencies across variants."
            )
        elif p_value < alpha_srm and not practical_breach:
            st.info(
                f"Statistically significant imbalance detected (p={p_value:.6f}), "
                f"but the largest deviation is only {max_rel_dev:.2%} — "
                f"below your practical threshold of {practical_threshold:.1%}. "
                "At your traffic scale this is expected. Groups are still comparable."
            )
        else:
            st.success("Split looks healthy. Safe to proceed with metric analysis.")
