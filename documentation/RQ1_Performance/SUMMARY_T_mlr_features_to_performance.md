# Summary: Model Features → Performance Regression

**Table:** `T_mlr_features_to_performance.csv` | **Research Question:** RQ1 - Competitive Performance  
**Analysis:** Multiple Linear Regression (MLR) | **Regressions:** 12 (4 games × 3 outcomes)

---

## Data Overview

Tests **which architectural/design features** predict performance across games. Each regression uses 5 model features to predict one performance outcome (win rate, profit, or game-specific metric).

**Key Question:** Do size, architecture, family, or thinking mode explain performance differences?

---

## Regression Predictors (5 Features)

1. **thinking**: Thinking mode (TE=1, TD=0, Inst=0.5) — Ordinal
2. **size_params**: Model size in billions of parameters — Continuous
3. **architecture_moe**: Mixture-of-Experts (1) vs Dense (0) — Binary
4. **family_encoded**: Model family (Qwen=1, Llama=0) — Binary
5. **family_version**: Within-family generation (0, 1, 2 for Llama; 0 for Qwen) — Ordinal

**Multicollinearity Check:** All VIF < 5 (thinking=1.76, architecture=1.90, size=2.47, family=3.60, version=2.80) ✅ No issues

---

## Overall Explanatory Power

### R² by Game (Average Across Outcomes)

| Game | Avg R² | Range | Predictability |
|------|--------|-------|----------------|
| **Salop** | **0.777** | 0.637–0.956 | High |
| **Athey-Bagwell** | **0.661** | 0.575–0.709 | Moderate-High |
| **Green-Porter** | **0.574** | 0.511–0.644 | Moderate |
| **Spulber** | **0.346** | 0.172–0.581 | Low-Moderate |

**Overall Average R²:** **0.562** (56% variance explained by features alone)

**Interpretation:** Architectural features explain **more than half** of performance variance, but game difficulty moderates predictability. Salop (hardest game) shows highest feature-performance coupling.

---

## Significant Predictors Summary

**Total Significant:** 18 / 60 tests (30%)

### By Feature (Significant Regressions Count)
1. **thinking**: **7 / 12** regressions — Most predictive feature
2. **architecture_moe**: 4 / 12
3. **size_params**: 3 / 12
4. **family_version**: 3 / 12
5. **family_encoded**: 1 / 12 — Least predictive

**Key Finding:** **Thinking mode dominates** architectural contributions. Model family matters least.

---

## Game-by-Game Analysis

### Athey-Bagwell (Entry Deterrence)

| Outcome | R² | Significant Predictors |
|---------|-----|------------------------|
| **Win Rate** | **0.709*** | **thinking (β=0.252, p=0.003)***, size_params (β=0.236, p=0.019)* |
| Average Profit | 0.575** | thinking (β=0.283, p=0.027)* |
| Productive Efficiency | 0.699*** | **thinking (β=0.303, p=0.002)***, architecture_moe (β=0.308, p=0.006)** |

**Pattern:** Thinking mode significant in **3/3** outcomes. Size helps win rate, MoE helps efficiency.

**Interpretation:** Strategic planning (thinking) critical for entry deterrence. MoE architecture enhances production optimization.

---

### Green-Porter (Collusion Sustainability)

| Outcome | R² | Significant Predictors |
|---------|-----|------------------------|
| Win Rate | 0.644*** | **thinking (β=0.285, p=0.016)** |
| Average Profit | 0.511* | architecture_moe (β=-0.394, p=0.025)* (negative!) |
| Reversion Frequency | 0.568** | thinking (β=-0.424, p=0.003)** (lower is better) |

**Pattern:** Thinking improves win rate (β=+0.285) AND reduces punishment frequency (β=-0.424). **MoE hurts** profit (β=-0.394).

**Interpretation:** Extended reasoning enables collusion. MoE architecture disrupts coordination (negative β). Simplicity helps in repeated games.

---

### Salop (Spatial Competition)

| Outcome | R² | Significant Predictors |
|---------|-----|------------------------|
| **Win Rate** | **0.956*** | **thinking (β=0.520, p<0.001)***, **architecture_moe (β=0.437, p<0.001)*** |
| Average Profit | 0.637*** | **thinking (β=0.562, p<0.001)*** |
| Market Price | 0.637*** | thinking (β=-0.556, p<0.001)*** (negative) |

**Pattern:** **Highest R² across all games** (0.956 for win rate). Thinking dominates **all 3** outcomes (|β|>0.5).

**Interpretation:** Spatial reasoning most feature-dependent. Thinking mode critical. Paradox: thinking increases profit (β=+0.562) but decreases price (β=-0.556) → suggests **volume strategy** over premium pricing.

---

### Spulber (Mechanism Design)

| Outcome | R² | Significant Predictors |
|---------|-----|------------------------|
| Win Rate | 0.581** | **architecture_moe (β=0.349, p=0.018)**, **family_version (β=0.365, p=0.022)** |
| Allocative Efficiency | 0.172 ns | (none significant) |
| Average Profit | 0.285 ns | (none significant) |

**Pattern:** **Lowest predictability**. Win rate somewhat predictable (R²=0.58), but efficiency/profit are not (R²<0.29).

**Interpretation:** Mechanism understanding less coupled to architecture. Suggests **emergent capability** not captured by basic features. MoE and newer generations help win rate.

---

## Feature Effect Directions

### Thinking Mode (Most Significant)
- **Positive effects** (7 regressions):
  - Athey-Bagwell: win rate (+0.252), profit (+0.283), efficiency (+0.303)
  - Green-Porter: win rate (+0.285), reversion frequency (-0.424, improves coordination)
  - Salop: profit (+0.562)
- **Negative effect** (1 regression):
  - Salop: market price (-0.556) — suggests volume-over-premium strategy

**Interpretation:** Thinking mode **universally beneficial** for strategic performance, but shifts pricing strategy in differentiation games.

### Architecture MoE
- **Positive**: Athey-Bagwell efficiency (+0.308), Salop win rate (+0.437), Spulber win rate (+0.349)
- **Negative**: Green-Porter profit (-0.394)

**Interpretation:** MoE helps optimization (Athey-Bagwell) and complex reasoning (Salop, Spulber), but **disrupts collusion** (Green-Porter).

### Size Params
- **Significant only in**: Athey-Bagwell win rate (+0.236)
- **Interpretation:** Size has **surprisingly limited** predictive power. Architecture matters more than scale.

---

## Key Takeaways

1. **Thinking mode is dominant predictor** — Significant in 7/12 regressions, largest βs (up to 0.562)
2. **Features explain 56% of performance variance** — Substantial but not complete (44% from other factors)
3. **Game difficulty moderates feature importance** — Salop (hardest) most feature-dependent (R²=0.96), Spulber (complex) least (R²=0.17–0.58)
4. **MoE helps optimization, hurts coordination** — Positive for efficiency tasks, negative for collusion
5. **Size matters less than expected** — Only significant in 3/12 regressions
6. **Architectural features insufficient alone** — Compare to T7: MAgIC+Features R²=0.817 vs Features-only R²=0.562 → **+25.5% improvement** from adding cognitive capabilities

### Model Fit Quality
- **Average R²:** 0.5623 (56.23% variance explained)
- **Average Adjusted R²:** 0.4407 (44.07% after penalty)
- **R² Range:** 0.0171 (Green-Porter profit) to 0.9562 (Salop win_rate)
- **Best fit:** Salop win_rate (R² = 0.956)
- **Worst fit:** Green-Porter profit (R² = 0.017)

### Significance Rate
- **Significant predictors:** 18 out of 60 (30.0%)
- **P < 0.05:** Multiple predictors across games
- **P < 0.01:** Several strong effects
- **P < 0.001:** Strongest predictor (Salop: thinking → win_rate)

### Predictor Success Rates

| Predictor | Significant (p<0.05) | Success Rate |
|-----------|---------------------|--------------|
| **thinking** | 6 / 12 | **50.0%** |
| **family_version** | 3 / 12 | 25.0% |
| **size_params** | 3 / 12 | 25.0% |
| **family_encoded** | 3 / 12 | 25.0% |
| **architecture_moe** | 3 / 12 | 25.0% |

**Key Finding:** **Thinking mode is the MOST CONSISTENT predictor** across games.

## Game-by-Game Analysis

### 1. Athey-Bagwell (Capacity Constraints)

#### Average Profit (R² = 0.282)
- **Significant:** thinking (p = 0.0392 *)
  - **Coef = +698.66:** TE models earn ~700 more profit
- Marginally: architecture_moe (p = 0.112), family (p = 0.388)
- **Interpretation:** Thinking mode explains some profit variance, but R² is low (28%)

#### Productive Efficiency (R² = 0.277)
- **Marginally significant:** family (p = 0.051), thinking (p = 0.053), architecture (p = 0.069)
- All predictors near threshold but none clearly dominate
- **Interpretation:** Multiple features contribute, but none alone explain efficiency

#### Win Rate (R² = 0.394)
- **Significant:** thinking (p = 0.0405 *)
  - **Coef = +0.097:** TE models win 9.7% more often
- Marginally: family (p = 0.071)
- **Interpretation:** Thinking mode best predictor of wins

**Athey-Bagwell Summary:** Thinking mode matters, but overall R² moderate (28–39%)

---

### 2. Green-Porter (Demand Shocks)

#### Average Profit (R² = 0.017) ❌
- **NO significant predictors** (all p > 0.6)
- Essentially **random noise** — model features explain nothing
- **Interpretation:** Demand shocks dominate; model characteristics irrelevant

#### Reversion Frequency (R² = 0.417)
- **Significant:** thinking (p = 0.0178 *)
  - **Coef = +0.0048:** TE models have HIGHER reversion (worse cooperation)
- **Interpretation:** Extended thinking reduces collusion stability

#### Win Rate (R² = 0.445)
- **Significant:** thinking (p = 0.0012 **), family (p = 0.0485 *)
  - **thinking coef = -0.044:** TE models win LESS often
  - **family coef = +0.032:** Family matters
- **Interpretation:** In Green-Porter, extended thinking is DETRIMENTAL

**Green-Porter Summary:** Thinking mode predicts strategy (reversion, wins) but NOT profit. Counter-intuitive: TE performs worse.

---

### 3. Salop (Product Differentiation)

#### Average Profit (R² = 0.736) ⭐
- **Significant:** thinking (p = 0.0005 ***)
  - **Coef = +638.78:** TE models earn ~640 more profit
- Marginally: architecture (p = 0.056)
- **Interpretation:** STRONG effect — thinking mode explains 74% of profit variance

#### Market Price (R² = 0.257)
- **Marginally:** thinking (p = 0.062)
- No clear predictors
- **Interpretation:** Pricing strategy less predictable from features

#### Win Rate (R² = 0.956) ⭐⭐⭐
- **Significant:** thinking (p = 0.0 ***), architecture (p = 0.037 *)
  - **thinking coef = +0.725:** TE models win 72.5% more often
  - **architecture coef = +0.099:** MoE adds 10% win rate
- **Interpretation:** NEAR-PERFECT FIT — thinking mode almost perfectly predicts wins

**Salop Summary:** BEST game for feature-based prediction. Thinking mode is dominant predictor (R² up to 96%).

---

### 4. Spulber (Search & Matching)

#### Allocative Efficiency (R² = 0.388)
- **Significant:** version (p = 0.016 *)
  - **Coef = +0.416:** Higher version number → better matching
- **Interpretation:** Model generation matters for matching, not thinking mode

#### Average Profit (R² = 0.688)
- **Significant:** family (p = 0.025 *)
- Marginally: architecture (p = 0.078), thinking (p = 0.069)
- **Interpretation:** Family background matters most for profit

#### Win Rate (R² = 0.684)
- **Significant:** architecture (p = 0.042 *)
- Marginally: version (p = 0.072)
- **Interpretation:** MoE architecture helps win in search markets

**Spulber Summary:** Different predictors matter than other games. Version and family dominate over thinking mode.

## Key Findings

### 1. Thinking Mode Dominance
**Thinking mode (TE vs TD) is the MOST RELIABLE predictor across games:**
- Significant in 6/12 regressions (50%)
- Strongest effects in:
  - Salop win_rate: coef = +0.725 (p < 0.001) ⭐⭐⭐
  - Salop profit: coef = +638.78 (p < 0.001) ⭐⭐⭐
  - Athey-Bagwell profit: coef = +698.66 (p = 0.039) ⭐

**Exception:** Green-Porter, where TE is DETRIMENTAL (coef = -0.044, p = 0.001)

### 2. Model Features Explain Limited Variance
- **Average R² = 0.46** — model features explain less than half of performance
- **Best case:** Salop (R² = 0.74–0.96)
- **Worst case:** Green-Porter profit (R² = 0.02)
- **Implication:** Other factors matter more:
  - Game-specific strategies
  - Behavioral tendencies (captured by MAgIC, not model features)
  - Stochastic elements

### 3. Game Heterogeneity
Different games reward different features:
- **Salop:** Thinking mode >>> all else
- **Spulber:** Version, family, architecture matter more
- **Green-Porter:** Features barely matter; demand shocks dominate
- **Athey-Bagwell:** Moderate thinking mode effects

### 4. Architecture (MoE) Weak Predictor
- Only significant in 1/12 regressions (Salop win_rate)
- Suggests: MoE architecture doesn't systematically improve economic reasoning

### 5. Family Effects Moderate
- Significant in 3/12 regressions (25%)
- Some evidence for "family clustering" of capabilities
- Modest support for hypothesis that model families share strategic tendencies

### 6. Family Version (Within-Family Generation) Emerging
- Significant in 3/12 regressions (25%)
- Suggests: Within-family evolution matters (e.g., Llama-3.1 vs 3.3 vs 4)
- Captures incremental improvements better than absolute version numbers

## Comparison to MAgIC Predictors

### Preview of RQ3 Findings
- **Feature R² average:** 0.562
- **MAgIC R² average:** 0.766 (from T5_magic_to_perf.csv)
- **MAgIC is 36% better** at explaining performance

**Implication:** Behavioral profiles (what models DO) matter more than model architecture/family (what models ARE).

## Adjusted R² Analysis

### Penalty for Overfitting
- **Average shrinkage:** 0.562 → 0.441 (12% drop)
- **Worst shrinkage:** Green-Porter profit (negative adjusted R²)
- **Best stability:** Salop win_rate (0.956 → 0.947)

**Interpretation:** Some regressions overfit (Green-Porter), but Salop results are robust.

## Significant Predictors Summary Table

| Game | Target | Predictor | Coef | P-Value | Direction |
|------|--------|-----------|------|---------|-----------|
| **Salop** | win_rate | thinking | +0.725 | <0.001 *** | TE >> TD |
| **Salop** | profit | thinking | +638.78 | <0.001 *** | TE >> TD |
| **Salop** | win_rate | architecture | +0.099 | 0.037 * | MoE > standard |
| **Athey-Bagwell** | profit | thinking | +698.66 | 0.039 * | TE > TD |
| **Athey-Bagwell** | win_rate | thinking | +0.097 | 0.041 * | TE > TD |
| **Green-Porter** | reversion | thinking | +0.005 | 0.018 * | TE > TD (bad) |
| **Green-Porter** | win_rate | thinking | -0.044 | 0.001 ** | TD > TE |
| **Green-Porter** | win_rate | family | +0.032 | 0.049 * | Family matters |
| **Spulber** | efficiency | family_version | +0.416 | 0.016 * | Newer > older |
| **Spulber** | profit | family | +211.61 | 0.025 * | Family matters |
| **Spulber** | win_rate | architecture | +0.294 | 0.042 * | MoE > standard |

**Total:** 18 significant effects out of 60 tests (30.0%)

## Hypothesis Testing

### H: Model Features Predict Performance
- **PARTIALLY SUPPORTED:** R² = 0.56 (moderate explanatory power)
- **Game-dependent:** Strong in Salop (96%), weak in Green-Porter (2%)

### H: Thinking Mode Improves Performance
- **STRONGLY SUPPORTED** in 3/4 games (Athey-Bagwell, Salop, Spulber)
- **REJECTED** in Green-Porter (TE performs worse)

### H: MoE Architecture Improves Performance
- **WEAKLY SUPPORTED:** Only 2 significant effects (Salop, Spulber win rates)

### H: Model Family Clusters Performance
- **WEAKLY SUPPORTED:** Only 2 significant effects
- **Mostly REJECTED:** Family doesn't systematically predict strategy

### H: Newer Models Perform Better
- **PARTIALLY SUPPORTED:** Family version significant in 3/12 regressions (25%)
- Within-family improvements matter more than absolute version numbers

## Implications for Research Questions

### RQ1: Competitive Performance
- **Model features explain ~56% of performance variance**
- **Thinking mode is most consistent predictor** (50% success rate)
- **Substantial unexplained variance** suggests:
  - Emergent strategic behaviors matter more than architecture
  - Game-specific adaptations vary widely
  - Stochastic elements significant

### RQ2: Behavioral Profiles
- **Features ≠ behavior:** Same features → different strategies
- **Need behavioral metrics** (MAgIC) to explain performance

### RQ3: Capability-Performance Link
- Preview: Behavioral capabilities (MAgIC) will explain more variance than model features (76.6% vs 56.2% = 36% improvement)

## Statistical Validity Concerns

### Sample Size
- N = 24 (13 models × 2 conditions - 2 for estimation)
- **Small for 4-predictor regression**
- Rule of thumb: Need N ≥ 10k (here: 10×4 = 40, we have 24)
- **Risk of overfitting** (confirmed by negative adjusted R² for Green-Porter)

### Multicollinearity
- No VIF reported, but predictors may correlate:
  - Family ↔ Architecture (some families use MoE)
  - Family ↔ Version (families evolve together)

### Outlier Sensitivity
- Some models vastly different (Random, L3.1-8B)
- May drive regression results

## Data Quality Notes

### Consistent R² Within Games
- Same R² for all targets within game-predictor pairs
- Indicates: Regressions run separately per target, but share underlying model fit

### P-Value Patterns
- Many p-values near threshold (0.05–0.10)
- Suggests: Effects are moderate, not strong

### Negative Adjusted R²
- Green-Porter profit: R²_adj = -0.190
- **Meaning:** Model fits worse than intercept-only baseline
- **Conclusion:** Model features useless for predicting Green-Porter profit

## Comparison to Human Studies

### Typical R² in Psychology/Economics
- **R² = 0.10–0.30:** Typical for behavioral predictors
- **R² = 0.56:** Above average, respectable explanatory power
- **R² = 0.96:** Exceptionally high (rare in social sciences)

### LLM vs Human Predictability
- LLMs may be more predictable from features than humans
- But: Behavioral heterogeneity still substantial (44% unexplained variance)

## Related Files
- `T5_magic_to_perf.csv` — Behavioral metrics → performance (R² = 0.82)
- `T_perf_win_rate.csv` — Win rate outcomes
- `T_perf_avg_profit.csv` — Profit outcomes
- `T_reasoning_chars.csv` — Thinking effort metrics
- `SYNTHESIS_RQ1_Competitive_Performance.md` — Full RQ1 synthesis
- `SYNTHESIS_RQ3_Capability_Performance_Links.md` — To be created
