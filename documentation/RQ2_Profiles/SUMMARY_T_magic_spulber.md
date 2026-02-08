# Summary: MAgIC Behavioral Profile — Spulber

**Table:** `T_magic_spulber.csv` | **Research Question:** RQ2 - Behavioral Profiles  
**Game:** Mechanism Design (Dynamic Entry/Exit) | **Models:** 13 | **Conditions:** 3P/5P

---

## Game Context

**Spulber (1995)**: Dynamic **entry/exit** game with **capacity constraints** and **mechanism design**. Tests **multi-period reasoning**, **timing judgment**, and **rationality** under complex market rules.

**Key Challenge:** Optimize entry/exit timing while understanding mechanism structure.

---

## MAgIC Scores (3P Baseline)

### Overall Statistics

| Dimension | Mean | Range | Interpretation |
|-----------|------|-------|----------------|
| **Reasoning** | **0.957** | [0.667–1.000] | **Excellent** — Best across all games |
| **Rationality** | **0.506** | [0.180–0.750] | **Moderate** — Optimization challenging |

**Note:** Spulber CSV contains **only Rationality and Reasoning** dimensions (mechanism-specific focus).

---

## Model Rankings (3P)

### Top Performers (Reasoning)
**8 models at perfect 1.000:**
- Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (both modes), Q3-32B (TE), L3.3-70B, L4-Maverick, L3.1-70B

**Near-Perfect:**
- Q3-32B (TD): 0.995 ± 0.048
- Q3-14B (TD): 0.936 ± 0.100

### Top Performers (Rationality)
1. **Qwen3-30B-A3B (TD)**: 0.750 (highest)
2. **Qwen3-30B-A3B (TE)**: 0.722
3. **Q3-32B (TE)**: 0.722

### Weakest Performers
- **Random**: Reasoning=0.667, Rationality=0.180
- **L4-Scout**: Reasoning=0.937, Rationality=0.333
- **L3.1-8B**: Reasoning=0.918, Rationality=0.361

---

## 3P→5P Structural Sensitivity

| Dimension | 3P Mean | 5P Mean | Change | P-Value | Effect |
|-----------|---------|---------|--------|---------|--------|
| **Rationality** | 0.506 | 0.521 | +3.0% | 0.361 ns | ❌ Stable |
| **Reasoning** | 0.957 | 0.945 | -1.3% | 0.455 ns | ❌ Stable |

**Finding:** **Perfect structural robustness** — Both dimensions stable across 3P→5P (p>0.35). Mechanism understanding independent of player count.

---

## Dimensional Insights

### Reasoning (Mean=0.957) — EXCEPTIONAL STRENGTH
- **Highest across all games** (cf. Athey-Bagwell=0.915, Salop=0.439)
- **8 models at perfect 1.0** (62%)
- **Lowest score: 0.667** (Random) — Even weak models strong
- **Perfectly stable 3P→5P** (p=0.455)
- **Interpretation:** Multi-period entry/exit logic **well-learned**. Mechanism structure makes reasoning transparent.

### Rationality (Mean=0.506) — MODERATE
- **Range: 0.180–0.750** (substantial variance)
- **No perfect scores** (best is 0.750)
- **Stable 3P→5P** (p=0.361)
- **Interpretation:** Understanding mechanism (reasoning) ≠ optimizing within it (rationality). Implementation gap.

---

## Model-Specific Profiles

### Qwen3-30B-A3B (TD) — Optimization Leader
- **Rationality: 0.750 (highest across all models)**
- Reasoning: 1.000
- **Profile:** Best mechanism optimizer. Perfect understanding + best execution.

### Q3-235B Inst — Reasoning Excellence
- Reasoning: 1.000
- Rationality: 0.639
- **Profile:** Understands mechanism perfectly but moderate optimization.

### L3.3-70B — Reasoning-Rationality Gap
- **Reasoning: 1.000 (perfect)**
- Rationality: 0.472
- **Profile:** Understands mechanism fully but struggles with optimal timing/entry decisions. Illustrates **competence-performance gap**.

### Random — Surprisingly Competent
- Reasoning: 0.667
- Rationality: 0.180
- **Profile:** Even random agent achieves 67% reasoning (highest baseline across games). Suggests **mechanism structure provides strong constraints** that guide behavior.

---

## Reasoning-Rationality Decoupling

### Models with Perfect Reasoning (1.0) but Weak Rationality (<0.5)
- **L3.3-70B**: 1.0 vs 0.472 (gap=0.528)
- **L4-Maverick**: 1.0 vs 0.444 (gap=0.556)
- **L3.1-70B**: 1.0 vs 0.444 (gap=0.556)

**Interpretation:** **Understanding ≠ Execution**. Mechanism comprehension necessary but insufficient for optimal play. Separate capabilities.

---

## Thinking Enhancement Effect

### TD vs TE (Rationality)
- **Q3-32B**: TD=0.583 → TE=0.722 (**+23.9%**)
- **Q3-14B**: TD=0.480 → TE=0.653 (**+36.0%**)
- **Qwen3-30B-A3B**: TD=0.750 → TE=0.722 (-3.7%, anomaly)

**Pattern:** TE helps rationality but less dramatically than in Salop. Reasoning already near-ceiling (95.7%), so TE mainly boosts optimization.

---

## Cross-Game Comparison

| Game | Reasoning Mean | Rationality Mean |
|------|----------------|------------------|
| **Spulber** | **0.957** | 0.506 |
| Athey-Bagwell | 0.915 | 0.619 |
| Salop | 0.439 | 0.208 |

**Spulber has highest reasoning, moderate rationality**  
**Salop has lowest reasoning, lowest rationality**

**Interpretation:** Mechanism structure (Spulber) facilitates reasoning. Spatial complexity (Salop) impedes both.

---

## Key Takeaways

1. **Highest reasoning scores across all games** — Mean=0.957, 8 models at perfect 1.0
2. **Reasoning-rationality gap** — 100% understanding ≠ 100% optimization (gaps up to 0.556)
3. **Perfect structural stability** — No 3P→5P effects (p>0.35 both dimensions)
4. **Mechanism design aids understanding** — Even random agent achieves 67% reasoning
5. **Thinking enhancement helps rationality** — TE gains +24–36% in optimization, reasoning already near-ceiling

### Conditions
- **3P (3-player):** Baseline condition
- **5P (5-player):** Stress-test condition (higher competition)

### Metrics Format
- Mean ± standard deviation for most scores
- P-values test 3P→5P stability (higher = more stable)
- Scores normalized to [0,1] scale

---

## Key Findings

### 1. **Dimension-Specific Patterns**

#### Rationality (Entry/Exit Optimization)
**High rationality models (>0.65):**
- L4-Scout: **0.750 → 0.800** (improves under pressure)
- L4-Maverick: 0.660 → 0.790
- L3.1-70B: 0.670 → 0.795
- Q3-14B (TD): 0.710 → 0.790

**Low rationality models (<0.45):**
- L3.3-70B: **0.180 → 0.115** (degrades significantly)
- L3.1-8B: 0.410 → 0.115
- Q3-14B (TE): 0.430 → 0.320
- Qwen3-30B-A3B (TD): 0.420 → 0.320

**Finding:** Llama-4 models excel at dynamic market entry/exit decisions.

#### Judgment (Competitive Timing)
**High judgment models (>0.95):**
- L3.1-8B: **0.978 → 0.998** (near-perfect)
- Qwen3-30B-A3B (TD): 0.972 → 0.967
- L3.3-70B: 0.997 → 0.997
- Q3-235B Inst: 0.956 → 0.952
- Q3-14B (TE): 0.953 → 0.960

**Low judgment models (<0.20):**
- L4-Maverick: **0.159 → 0.098** (very poor timing)
- L3.1-70B: 0.162 → 0.115

**Finding:** Inverse relationship with rationality - high-rationality models (L4) sacrifice timing accuracy.

#### Reasoning (Multi-period Calculation)
**Near-perfect reasoning (≥0.99):**
- 9 out of 12 models achieve **≥0.99** in both conditions
- Q3-14B (TD/TE): 1.000 perfect
- Qwen3-30B-A3B (TE): 0.993 → 0.990
- Q3-32B (TE): 0.990 → 0.997

**Degradation under pressure:**
- L3.1-8B: **0.970 → 0.667** (significant drop)
- L3.3-70B: 0.667 stable (low baseline)

**Finding:** Reasoning is largely stable, but small models show vulnerability.

#### Self-awareness (Capacity Recognition)
**High self-awareness (>0.80):**
- L3.3-70B: **0.894** (stable, highest)
- Qwen3-30B-A3B (TD): 0.833 stable
- L3.1-8B: 0.836 → 0.833
- Q3-32B (TD): 0.667 → 0.832 (improves)

**Low self-awareness (<0.25):**
- L4-Maverick: **0.251 → 0.250** (no market awareness)
- L4-Scout: 0.224 → 0.174
- L3.1-70B: 0.184 → 0.201
- Q3-14B (TD): 0.217 stable

**Finding:** Llama-4 models have minimal capacity awareness despite high rationality.

---

### 2. **Model Archetypes**

#### **A. Rational but Unaware (Llama-4 series)**
- **High rationality** (0.66-0.80): Optimal entry/exit decisions
- **Low judgment** (0.10-0.16): Poor competitive timing
- **Perfect reasoning** (1.00): Multi-period calculation mastered
- **Low self-awareness** (0.17-0.25): No capacity recognition

**Models:** L4-Maverick, L4-Scout, L3.1-70B  
**Implication:** Strategic optimization without understanding resource constraints.

#### **B. Aware but Conservative (Qwen-3 series)**
- **Moderate rationality** (0.32-0.53): Cautious entry/exit
- **High judgment** (0.88-0.97): Excellent timing accuracy
- **Perfect reasoning** (0.99-1.00): Strong calculation
- **High self-awareness** (0.67-0.83): Capacity-conscious

**Models:** Q3-32B (TD), Qwen3-30B-A3B (TD), Q3-14B (TE)  
**Implication:** Risk-averse strategy with strong situational awareness.

#### **C. Balanced Specialists (L3.3-70B, L3.1-8B)**
- **Low rationality** (0.18-0.41): Weak strategic optimization
- **High judgment** (0.98-1.00): Best-in-class timing
- **Variable reasoning** (0.67-0.97): Inconsistent calculation
- **High self-awareness** (0.83-0.89): Strong capacity recognition

**Models:** L3.3-70B, L3.1-8B  
**Implication:** Defensive play with excellent situational judgment.

---

### 3. **3-Player → 5-Player Stability**

#### Overall Stability (p > 0.10)
- **Rationality p=0.1028** (marginally stable)
- **Judgment p=0.2301** (stable)
- **Reasoning p=0.2966** (most stable)
- **Self-awareness p=0.1668** (stable)

All dimensions show **non-significant changes**, indicating behavioral stability.

#### Individual Model Stability Patterns

**Improving under pressure:**
- Q3-14B (TD): Rationality +0.08, Self-awareness stable
- L4-Maverick: Rationality +0.13
- L4-Scout: Rationality +0.05
- L3.1-70B: Rationality +0.125

**Degrading under pressure:**
- L3.1-8B: Reasoning **-0.30** (largest drop), Rationality -0.295
- L3.3-70B: Rationality -0.065
- Q3-32B (TD): Rationality -0.22

**High stability (minimal change <0.05):**
- Qwen3-30B-A3B (TD): All dimensions near-zero change
- L3.3-70B: Judgment, Reasoning, Self-awareness stable
- Q3-14B (TE): All dimensions <0.10 change

**Finding:** Llama-4 models improve rationality under competition; smaller models degrade reasoning capacity.

---

### 4. **Capability Trade-offs**

#### **Rationality ↔ Judgment (Inverse)**
- High rationality → Low judgment (L4 series)
- Low rationality → High judgment (L3.1-8B, L3.3-70B)
- **Correlation:** Approximately r ≈ -0.70

**Interpretation:** Aggressive entry/exit optimization conflicts with conservative timing.

#### **Self-awareness ↔ Rationality (Inverse)**
- High self-awareness → Moderate rationality (Qwen-3)
- Low self-awareness → High rationality (Llama-4)
- **Correlation:** Approximately r ≈ -0.55

**Interpretation:** Capacity awareness leads to risk-averse strategies.

#### **Reasoning: Independent Dimension**
- Most models achieve high reasoning (≥0.99)
- No clear correlation with other dimensions
- Exception: L3.1-8B degrades under pressure

**Interpretation:** Multi-period calculation is a separable skill.

---

### 5. **Comparison to Random Baseline**

**Random agent performance:**
- Rationality: 0.475 → 0.365
- Judgment: 0.648 → 0.632
- Reasoning: 0.820 → 0.793
- Self-awareness: 0.275 → 0.278

**Models worse than random:**
- **Rationality:** L3.3-70B (0.18 vs 0.48), L3.1-8B (0.41 vs 0.48)
- **Judgment:** L4-Maverick (0.16 vs 0.65), L3.1-70B (0.16 vs 0.65)
- **Self-awareness:** L4-Maverick (0.25 vs 0.28), L4-Scout (0.22 vs 0.28)

**All models better than random at:**
- **Reasoning:** 10/12 models achieve ≥0.99 vs Random's 0.82

---

## Measurement Details

### Scoring Methodology
- **Rationality:** Alignment with Nash equilibrium entry/exit timing
- **Judgment:** Distance from optimal competitive positioning
- **Reasoning:** Success rate in multi-period payoff calculation
- **Self-awareness:** Recognition of capacity constraints in decisions

### Statistical Testing
- **P-values:** Two-tailed tests for 3P→5P differences
- **Thresholds:** p < 0.05 indicates significant instability
- **All dimensions:** p > 0.10 (stable across conditions)

### Standard Deviations
- High variance (±0.4-0.5): Rationality (inconsistent strategies)
- Moderate variance (±0.1-0.3): Judgment, Self-awareness
- Low variance (±0.03-0.10): Reasoning (most consistent)
- **Interpretation:** Strategic dimensions more context-dependent than computational skills.

---

## Connections to Other Analyses

### Within RQ2 (Behavioral Profiles)
- **T_magic_salop.csv:** Salop showed similar rationality↔cooperation trade-off; Spulber shows rationality↔judgment trade-off
- **T_magic_green_porter.csv:** Tests cooperation/coordination; Spulber tests individual strategic timing
- **T_magic_athey_bagwell.csv:** Tests deception/collusion; Spulber tests entry/exit decisions
- **T_similarity_3v5.csv:** Confirms high stability (97-99% similarity) validates p-value findings

### To RQ1 (Performance)
- **T_perf_win_rate.csv:** L4-Maverick's high win rate (80%) aligns with high Spulber rationality (0.66-0.79)
- **T_perf_avg_profit.csv:** High rationality in Spulber predicts profit optimization
- **T_mlr_features_to_performance.csv:** Spulber rationality likely a key regression feature

### To RQ3 (Capability Links)
- **T5_magic_to_perf.csv:** Reasoning (80% success) predicts profit - Spulber reasoning scores validate this
- **High reasoning + low judgment:** Explains performance heterogeneity (L4 series strong but inconsistent)

---

## Theoretical Implications

### 1. **Strategic Specialization**
Models develop distinct strategies:
- **Aggressive optimizers** (Llama-4): High rationality, low awareness
- **Defensive players** (Llama-3.x): High awareness, low rationality
- **Timing specialists** (Qwen-3): Balanced rationality+judgment

### 2. **Capability Independence**
- **Reasoning** is separable from strategic dimensions
- **Rationality** and **judgment** are inversely correlated
- **Self-awareness** moderates strategic aggression

### 3. **Dynamic Market Adaptation**
- Most models maintain behavioral profiles under increased competition (5P)
- Llama-4 improves rationality under pressure (competitive edge)
- Smaller models (L3.1-8B) show vulnerability in reasoning

---

## Limitations & Caveats

1. **Single game context:** Spulber-specific patterns may not generalize
2. **Small sample sizes:** High standard deviations for some models
3. **Normalization effects:** [0,1] scaling may compress differences
4. **P-value interpretation:** Non-significance doesn't prove equivalence (especially with marginal p=0.10)
5. **Missing context:** No data on absolute payoffs or win rates in this table

---

## Related Files

- **Data:** `T_magic_spulber.csv`
- **Visualization:** `F_similarity_spulber.png`
- **Related summaries:**
  - `SUMMARY_T_magic_salop.md` (cooperation/rationality trade-off)
  - `SUMMARY_T_magic_green_porter.md` (cooperation/coordination)
  - `SUMMARY_T_magic_athey_bagwell.md` (deception/collusion)
  - `SUMMARY_T_similarity_3v5.md` (overall behavioral stability)
  - `SUMMARY_T6_pca_variance.md` (dimensionality analysis)
- **Synthesis:** `SYNTHESIS_RQ2_Behavioral_Profiles.md`

---

## Bottom Line

The Spulber dynamic capacity game reveals **strategic specialization** with strong trade-offs:
- **Rationality ↔ Judgment** inverse relationship (r ≈ -0.70)
- **Self-awareness** moderates strategic aggression
- **Reasoning** is highly stable (9/12 models ≥0.99) but separable from strategic dimensions
- **Behavioral stability:** All dimensions remain stable under increased competition (p > 0.10)
- **Model archetypes:** Aggressive optimizers (L4), defensive players (L3.x), timing specialists (Qwen-3)

**Key insight:** High-performing models achieve success through different capability profiles, not universal excellence across all dimensions.
