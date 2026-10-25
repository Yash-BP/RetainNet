# RetainNet — Complete Engineering Specification Suite

**Project Name:** RetainNet: Deep Survival Analysis Framework for Dynamic Customer Lifetime Value (CLTV) & Proactive Churn Mitigation  
**Version:** 1.0.0 (Production Baseline)  
**Target Codebase:** `retainnet/` Python Package, `main.py` CLI Engine, `dashboard.py` Enterprise Intelligence System, `tests/` (119 Automated Verification Tests)

---

# 4. UI/UX Specification Document

## 4.1 Design Principles & Aesthetic Direction
RetainNet's interface follows an **Editorial Financial & Clinical Intelligence** aesthetic inspired by institutional quantitative platforms:
1. **Warm Paper Canvas over Harsh White/Dark Neon:** Uses a warm parchment background (`#F5F4ED`) paired with crisp white cards (`#FFFFFF`), deep pine-charcoal typography (`#181E19`), and muted stone borders (`#E5E4D8`) to reduce eye strain during long analytical sessions.
2. **Semantic Risk Color Encoding:** Every chart, badge, and metric uses a strict 3-tier semantic palette:
   - **Forest Green (`#3B5740` / `#EAF1EC`):** Healthy retention, high CLTV, best-performing model, and positive counterfactual lift.
   - **Warm Amber (`#C38438` / `#FAF3E8`):** Moderate priority, price-sensitive segment, and synthetic demonstration notices.
   - **Terracotta Red (`#BD6448` / `#F8ECE7`):** Critical churn urgency, at-risk revenue exposure, and negative campaign ROI.
3. **Zero-Flicker Instant Interactivity:** All 6 navigation views, dataset switches, subscriber searches, and What-If slider adjustments execute via client-side DOM and Chart.js updates in $< 50\text{ ms}$ with `chart.update('none')`.

## 4.2 Typography, Color Palette & Spacing Tokens

### 4.2.1 Typography Scale
- **Primary Font Family:** `'Plus Jakarta Sans', 'Inter', sans-serif` (loaded via Google Fonts with system sans fallback).
- **Monospace Font Family:** `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace` (used for mathematical formulas and column names).
- **Scale & Weights:**
  - Page Header (`#page-heading`): `1.25rem` (`20px`), `font-weight: 800` (`extrabold`), tracking `-0.025em`.
  - KPI Primary Value (`.kpi-value`): `1.75rem` (`28px`) desktop / `1.30rem` (`20.8px`) mobile, `font-weight: 800`, line-height `1.15`.
  - Card Section Heading (`h3`, `h4`): `1.0rem` (`16px`), `font-weight: 700` (`bold`).
  - KPI Label (`.kpi-label`): `0.72rem` (`11.5px`), `font-weight: 700`, uppercase, letter-spacing `0.06em`, color `#6C7567`.
  - Body / Table Cell / Control Copy: `0.75rem` (`12px`), `font-weight: 500`–`600`.

### 4.2.2 Color Token Specification

| Token Name | Hex Code | Usage Context | WCAG Contrast vs. Background |
| :--- | :--- | :--- | :--- |
| `canvas` | `#F5F4ED` | Main application background | — |
| `sidebar-bg` | `#ECEBE3` | Left navigation & global parameter sidebar | — |
| `cardbg` | `#FFFFFF` | KPI cards, chart containers, data tables | — |
| `branddark` | `#181E19` | Primary headings, active nav pill, KPI values | `16.4 : 1` on `#FFFFFF` (AAA) |
| `charcoal` | `#222621` | Primary table text, form inputs, descriptions | `14.8 : 1` on `#FFFFFF` (AAA) |
| `subtle` | `#6A7265` | Secondary labels, axis ticks, helper text | `4.8 : 1` on `#FFFFFF` (AA) |
| `forest` | `#3B5740` | Primary brand accent, DeepSurv curves, healthy badges | `7.9 : 1` on `#FFFFFF` (AAA) |
| `terracotta` | `#BD6448` | High-risk bars, baseline hazard curves, critical alerts | `4.6 : 1` on `#FFFFFF` (AA) |
| `ambercustom` | `#C38438` | Moderate risk bars, Stage 2 badge, Zero-Shot badge | Used with dark text on light tint |
| `lineborder` | `#E5E4D8` | Card borders (`1px solid`) and table row dividers | — |

### 4.2.3 Layout & Spacing System
- **Desktop Viewport (`> 1024px`):** Fixed two-column split (`h-screen w-screen overflow-hidden`). Left sidebar fixed at `288px` (`w-72`) with independent vertical scroll; main workspace (`flex-1`) with sticky top header bar and scrollable content container (`max-w-7xl mx-auto p-8 space-y-8 pb-32`).
- **Card Geometry:** `border-radius: 1rem` (`16px`) for KPI and content cards, `1px solid #E6E4D9` border, `padding: 1.25rem 1.35rem` (`20px 21.6px`), subtle elevation shadow `0 1px 3px rgba(0,0,0,0.02)`.

## 4.3 Navigation & Screen-by-Screen Specifications

### 4.3.1 Persistent Left Sidebar (`aside`)
1. **Brand Header:** `RN` monogram badge (`40x40px` `#181E19` rounded-xl) + `RetainNet` title + `Enterprise Intelligence` subtitle.
2. **Workspace Indicator:** `Northstar Production` card with pulsing green status dot (`w-2 h-2 rounded-full bg-forest animate-pulse`).
3. **6-Tab Navigation Menu (`nav`):**
   - `1. Executive Summary` (`#tab-btn-overview`)
   - `2. Pipeline & Benchmarks` (`#tab-btn-pipeline`)
   - `3. Customer Prescriptions` (`#tab-btn-customers`)
   - `4. What-If Simulator` (`#tab-btn-simulator`)
   - `5. MLOps & DriftLens` (`#tab-btn-mlops`)
   - `6. Audit Reports & Export` (`#tab-btn-reports`)
4. **Global Parameters Panel:**
   - **Data Source Selector (`#dataset-selector`):** Switches between `IBM Telco Benchmark (2,500)`, `Synthetic Weibull Cohort (2,500)`, and `Upload Custom CSV`.
   - **Conditional Box 1 (`#synthetic-controls-box`):** Revealed when `Synthetic Weibull Cohort` is selected. Contains `Sample Size` (`50–10,000`), `Random Seed` (`1–99,999`), `Regenerate Cohort` button, and active record counter.
   - **Conditional Box 2 (`#custom-csv-upload-box`):** Revealed when `Upload Custom CSV` is selected. Contains `CSV template` download link, drag-and-drop dropzone (`#csv-dropzone`), validation status banner (`#csv-validation-feedback`), and the interactive **Auto-Adapted Columns Schema Mapper** (`#csv-schema-mapper`) with 4 dropdowns (`Value / Sales ($)`, `Category / Tier`, `Record / Cust ID`, `Tenure / Date`).
   - **Inference Engine Selector (`#model-selector`):** Switches active customer survival curves and leaderboard highlights between `RetainNet DeepSurv (PyTorch)`, `DeepHit Discrete Engine`, `CoxPH Regularized Baseline`, and `Random Survival Forest (RSF)`.
   - **Forecast Horizon Slider (`#horizon-slider`):** Range `12` to `72` months (`step=6`), dynamically rescaling time axes across Tabs 2, 3, 4, and 6.
   - **Financial Inputs (`#cac-input`, `#discount-input`):** Numeric inputs for `CAC ($)` (default `120`) and `Discount Rate (d)` (default `0.01`, step `0.005`), recalculating portfolio and customer CLTV in real time.
5. **Live Backend Status Footer:** Displays active dataset source, subscriber count, and `Pipeline Operational` indicator.

### 4.3.2 Screen 1: Executive Summary & Portfolio Health (`#tab-overview`)
- **Conditional Banner (`#synthetic-demo-banner`):** Amber notice bar shown only when the Synthetic Weibull Cohort is active.
- **4-Column KPI Grid:** `Active Subscribers`, `Average Dynamic CLTV`, `At-Risk Revenue` (terracotta value), and `12-Mo Projected Churn`.
- **12-Column Visual Grid:**
  - **Left (7 cols):** `Dynamic CLTV Distribution` bar chart (`#cltvHistogramChart`) with 10 bins color-coded by risk tier (`2` terracotta, `2` amber, `6` forest green).
  - **Right (5 cols):** `K-Means Stage 1 Archetypes` doughnut chart (`#archetypesDonutChart`, `72%` cutout) with 2x2 percentage legend.
- **Priority Customer Queue Table:** Displays the top 6 highest-risk subscribers (`Customer ID`, `Segment`, `Monthly Fee`, `Dynamic LTV`, `12M Survival`, `Risk Status` pill badge, and `Prescribe` action button that navigates directly to Tab 3 for that subscriber).

### 4.3.3 Screen 2: 3-Stage Pipeline & Model Benchmarking (`#tab-pipeline`)
- **Conditional Notice (`#pipeline-inference-notice`):** Blue informational banner shown when an unlabeled custom CSV is active, explaining why ground-truth evaluation metrics are marked `N/A (Inference-Only)`.
- **3-Column KPI Grid:** `Top Architecture`, `Harrell's C-Index`, and `Denoised Signal Retention`.
- **3-Stage Denoising Funnel Cards:** Side-by-side architectural breakdown of Stage 1 (`X_aug = [X_raw | OneHot_Clusters]`), Stage 2 (`0.30 <= p_hat <= 0.70` excision), and Stage 3 (`Loss = -sum [h_theta(x_i) - log sum exp(h_theta(x_j))]`).
- **Benchmark Comparison Leaderboard Table:** Compares all 4 models across `Harrell's C-Index`, `Integrated Brier Score`, `BS @ 12M`, `BS @ 24M`, `BS @ 36M`, and `D-Calibration Chi2`, highlighting the top model with a `BEST` badge.
- **Kaplan-Meier vs. Predicted DeepSurv Chart (`#pipelineKmChart`):** Dual-line continuous trajectory comparison (`Predicted DeepSurv` solid forest line with area fill vs. `Empirical Kaplan-Meier` dashed slate line).

### 4.3.4 Screen 3: Customer Lookup & Clinical Prescriptions (`#tab-customers`)
- **Subscriber Search & Selection Bar:**
  - Real-time search input (`#customer-search-input`, filters by ID, segment, urgency, or contract).
  - Full-cohort dropdown (`#customer-dropdown`) populated via `DocumentFragment` containing all `2,500+` subscribers.
  - Match count badge (`#customer-count-badge`) and empty-state alert (`#customer-no-match-msg`).
  - Subscriber metadata pills (`Tenure`, `Monthly ARPU`, `Contract`).
- **Individualized Survival Function Chart (`#customerConformalChart`):** Plots the subscriber's point estimate $\hat{S}(t \mid \mathbf{x})$ surrounded by shaded 90% CiPOT conformal upper and lower bounds.
- **Retention Intervention Recommendation Card:** Displays urgency badge (`Critical Urgency` / `Moderate Priority` / `Healthy / Low Risk`), clinical prescription narrative, `Current Hazard Rate`, `Projected Dynamic LTV`, and `Intervention ROI`.

### 4.3.5 Screen 4: What-If Retention Simulator (`#tab-simulator`)
- **Left Control Panel (4 cols):**
  - `Monthly Bill Discount` slider (`$0` to `$35 / mo`, step `$5`, clamped at `$30` effective max) with live $\gamma_{\text{disc}}$ label.
  - `12-Month Lock-in Contract` toggle switch ($\gamma = 0.60$).
  - `VIP Success Concierge` toggle switch (`$25` one-time cost, $\gamma = 0.85$).
  - `Effective Hazard Ratio (γ)` summary box displaying composite $\gamma$ and `% Churn Risk` reduction.
- **Right Simulation Panel (8 cols):**
  - `Baseline vs Counterfactual Survival Curve` (`#counterfactualChart`) comparing unassisted $S(t)$ (terracotta) vs. promoted $S^*(t) = S(t)^\gamma$ (forest green shaded).
  - 4-Card Financial Impact Row: `Baseline LTV`, `Counterfactual LTV`, `Net LTV Lift` (green if positive, terracotta if negative), and `Campaign ROI %`.

### 4.3.6 Screen 5: MLOps & DriftLens (`#tab-mlops`)
- **3-Column KPI Grid:** `Population Stability Index (PSI)`, `Kolmogorov-Smirnov (KS) Test`, and `Conformal Coverage @ 90%`.
- **Left Panel (7 cols):** `DriftLens Feature Distribution Stability` grouped bar chart (`#driftChart`) comparing Reference Training Cohort vs. Active Evaluation Cohort across 5 core features, paired with a dynamic `STABLE` / `DRIFT DETECTED` status badge.
- **Right Panel (5 cols):** `SurvivalPFN Cold-Start Engine` form (`Contract Type`, `Monthly Charges ($)`, `Internet Service`, and `Generate Prior Survival Distribution` button) outputting the matched K-Means archetype prior and expected 12-month retention probability.

### 4.3.7 Screen 6: Audit Reports & Export (`#tab-reports`)
- **5-Column Audit Summary Row:** `CUSTOMERS`, `RIGHT-CENSORED`, `BEST C-INDEX`, `SIGNAL RETAINED`, and `FORECAST`.
- **Export & Compliance Cards:** Primary download CTA (`Download customer LTV predictions` `.csv`) with dynamic horizon and subscriber count descriptions, paired with the `Audit status` compliance card.
- **Model Evaluation Audit Table:** Full multi-horizon Brier Score and C-Index audit record.

## 4.4 Loading, Empty, Error & Responsive States
- **Loading State:** Streamlit displays `"Training RetainNet survival models and initializing Enterprise Intelligence Engine..."` on cold start while `@st.cache_resource` executes; subsequent interactions are instantaneous.
- **Empty Search State:** When `#customer-search-input` matches zero records, `#customer-dropdown` disables and `#customer-no-match-msg` displays `"No matching subscribers found."` in terracotta text while preserving the last rendered chart.
- **Error State (Invalid CSV Upload):** Displays a red validation box (`bg-red-50 border-red-200 text-red-900`) inside `#csv-validation-feedback` stating the exact rejection reason (e.g., `"Rejected (bad.csv): Invalid monthly_charges '-50' on row 2. Must be > 0."`) while keeping the previous valid dataset active.
- **Responsive Breakpoints:**
  - **Tablet (`<= 1024px`):** Sidebar transitions from fixed left column to top full-width header; navigation links reflow into a 3-column grid; KPI cards reflow to 2 columns; tables enable smooth horizontal touch scrolling (`min-width: 580px`).
  - **Mobile (`<= 640px`):** Navigation links reflow into a compact 2-column grid (`min-height: 38px`); KPI values scale to `1.3rem`; header elements stack vertically.
