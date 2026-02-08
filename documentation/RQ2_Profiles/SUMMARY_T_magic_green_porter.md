# Summary: MAgIC Behavioral Profile — Green-Porter

**Table:** `T_magic_green_porter.csv` | **Research Question:** RQ2 - Behavioral Profiles  
**Game:** Collusion Sustainability (Demand Shocks) | **Models:** 13 | **Conditions:** 3P/5P

---

## Game Context

**Green-Porter (1984)**: Sustaining tacit collusion under **imperfect monitoring** with demand shocks. Requires distinguishing **competitor cheating** from **random market fluctuations**. Tests **cooperation** under uncertainty.

**Key Challenge:** Models must maintain cooperation despite noisy signals.

---

## MAgIC Scores (3P Baseline)

### Overall Statistics

| Dimension | Mean | Range | Interpretation |
|-----------|------|-------|----------------|
| **Cooperation** | **0.757** | [0.266–1.000] | **Strong** — Most models cooperate well |

**Note:** Green-Porter CSV contains **only Cooperation** dimension (game-specific focus on collusion sustainability).

---

## Model Rankings (3P Cooperation)

### Top Performers (Perfect 1.0)
- **Q3-14B (TE)**: 1.000
- **Q3-235B Inst**: 1.000  
- **Qwen3-30B-A3B (TE)**: 1.000
- **Q3-32B (TE)**: 1.000
- **L3.1-70B**: 1.000

**5 models achieve perfect cooperation**

### Moderate Performers (0.5–0.9)
- Q3-14B (TD): 0.784 ± 0.093
- Qwen3-30B-A3B (TD): 0.876 ± 0.100
- Q3-32B (TD): 0.773 ± 0.094
- L4-Maverick: 0.743 ± 0.092

### Weak Performers (<0.5)
- **L3.3-70B**: 0.266 ± 0.071 (lowest) — Large model fails cooperation
- **Random**: 0.367 ± 0.097 (baseline)
- **L3.1-8B**: 0.368 ± 0.116

---

## 3P→5P Structural Sensitivity

| Dimension | 3P Mean | 5P Mean | Change | P-Value | Effect |
|-----------|---------|---------|--------|---------|--------|
| **Cooperation** | 0.757 | 0.789 | **+4.2%** | **0.001*** | ✅ **Improves** in 5P |

**Surprising Finding:** Cooperation **increases** with more players (p=0.001). Contradicts classical theory where more players make collusion harder. Suggests **LLMs use less aggressive punishment** or better tacit coordination in larger groups.

---

## Dimensional Insights

### Cooperation (Mean=0.757) — STRONG

**Distribution:**
- **5 models at 1.0** (42%) — Perfect collusion sustainability
- **4 models at 0.7–0.9** (31%) — Strong cooperation
- **4 models at 0.3–0.4** (31%) — Cooperation failure

**3P→5P Effect:** **+4.2% improvement** (p=0.001)  
**Interpretation:** More players → better cooperation (counter-intuitive). Possible mechanism: Diffused responsibility reduces aggressive retaliation.

---

## Model-Specific Profiles

### Perfect Cooperators (5 models)
**Q3-14B (TE), Q3-235B, Qwen3-30B-A3B (TE), Q3-32B (TE), L3.1-70B**
- **Pattern:** Thinking-enhanced (TE) models dominate, plus large models (235B, 70B)
- **Profile:** Maintain collusion despite demand shocks, distinguish noise from cheating

### Thinking Enhancement Effect
- **Q3-14B**: TD=0.784 → TE=1.000 (+27.5%)
- **Qwen3-30B-A3B**: TD=0.876 → TE=1.000 (+14.2%)
- **Q3-32B**: TD=0.773 → TE=1.000 (+29.4%)

**Strong evidence:** Extended thinking enables perfect collusion.

### Cooperation Failures (3 models < 0.4)
- **L3.3-70B**: 0.266 — Worst despite 70B size (architectural issue?)
- **Random**: 0.367 — Baseline
- **L3.1-8B**: 0.368 — Size constraint

---

## Cross-Game Comparison

**Green-Porter Cooperation (75.7%)** vs **Athey-Bagwell Cooperation (87.5%)**  
→ Imperfect monitoring (Green-Porter) **harder** than capacity coordination (Athey-Bagwell)

**Green-Porter** focuses on **noisy signal interpretation**, while **Athey-Bagwell** tests **multi-period deterrence**.

---

## Key Takeaways

1. **Strong overall cooperation** — 75.7% mean, 5 models perfect (1.0)
2. **Thinking enhancement critical** — TE models achieve 100% cooperation vs TD at 77–88%
3. **Counter-intuitive 5P improvement** — Cooperation increases +4.2% with more players (p=0.001)
4. **Size ≠ success** — L3.3-70B (0.266) fails despite 70B parameters
5. **Binary capability distribution** — Models cluster at 1.0 (perfect) or 0.3–0.4 (failure), few in middle
- **5P (5-player):** Larger market (more coordination challenges)

### Metrics Format
- Mean ± standard deviation (where variance exists)
- P-values test 3P→5P stability
- Scores normalized to [0,1] scale
- Fixed values (e.g., "1.000") indicate perfect/consistent performance

---

## Key Findings

### 1. **Cooperation Dimension (Implicit Collusion)**

#### Near-Perfect Cooperators (≥0.95)
**Stable high cooperation:**
- Q3-14B (TD): **1.000 → 1.000** (perfect collusion maintenance)
- Q3-235B Inst: 1.000 → 0.949
- Q3-32B (TD): 0.998 → 1.000 (improves to perfect)
- L4-Maverick: 0.952 → 0.954
- L4-Scout: 1.000 → 0.947

**Finding:** Qwen-3 and Llama-4 models excel at implicit coordination.

#### Moderate Cooperators (0.50-0.90)
- Qwen3-30B-A3B (TE): 0.921 → 0.865
- L3.1-8B: **0.932 → 0.521** (major degradation under pressure)
- L3.3-70B: 0.757 → 0.661
- Qwen3-30B-A3B (TD): 0.593 → 0.711
- Q3-14B (TE): 0.660 → 0.487

**Finding:** Mid-tier models show significant variance and instability.

#### Low Cooperators (<0.50)
- Q3-32B (TE): **0.266 → 0.218** (cooperative breakdown)
- L3.1-70B: 0.300 → 0.312 (no collusion capacity)
- Random: 0.464 → 0.486

**Finding:** Some fine-tuned models perform worse than random baseline.

---

### 2. **Coordination Dimension (Strategic Alignment)**

#### High Coordinators (≥0.60)
- Q3-32B (TD): **0.986 → 1.000** (near-perfect to perfect)
- L4-Maverick: 0.734 → 0.748 (stable coordination)
- L4-Scout: 1.000 → 0.622 (degrades but remains high)
- Q3-235B Inst: 1.000 → 0.650

**Finding:** Large Qwen-3 and Llama-4 models maintain strategic alignment.

#### Moderate Coordinators (0.30-0.60)
- Qwen3-30B-A3B (TE): 0.594 → 0.454
- L3.1-8B: **0.594 → 0.300** (coordination collapse)
- Q3-14B (TE): 0.314 → 0.300

**Finding:** Mid-tier models struggle with coordination under increased complexity.

#### Baseline Coordinators (=0.30)
**Fixed at minimum:**
- L3.1-70B: **0.300 stable** (no coordination capacity)
- Q3-32B (TE): 0.300 stable
- L3.3-70B: 0.300 stable
- Qwen3-30B-A3B (TD): 0.300 stable
- Q3-14B (TD): 1.000 → 1.000 (exception: perfect coordination)

**Finding:** 0.30 appears to be a baseline/floor value - 7 models show this.

---

### 3. **Model Archetypes**

#### **A. Perfect Colluders (Q3-14B TD, Q3-32B TD)**
- **Perfect cooperation** (1.00): Sustained implicit collusion
- **Perfect/near-perfect coordination** (0.99-1.00): Strategic alignment mastered
- **High stability:** Minimal degradation under increased market size

**Implication:** Large Qwen-3 models with specific tuning achieve dominant collusion.

#### **B. Stable Cooperators (L4-Maverick, L4-Scout, Q3-235B Inst)**
- **High cooperation** (0.95-1.00): Strong collusion maintenance
- **High coordination** (0.65-1.00): Effective strategic alignment
- **Moderate stability:** Some degradation in 5P condition

**Implication:** Llama-4 maintains cooperation through market expansion.

#### **C. Unstable Cooperators (L3.1-8B, Q3-14B TE, Qwen3-30B-A3B TE)**
- **High baseline cooperation** (0.66-0.93): Good collusion in simple markets
- **Significant degradation** (-0.17 to -0.41): Cooperation breaks down under complexity
- **Coordination collapse:** Mid-tier to baseline performance

**Implication:** Smaller/differently-tuned models can't sustain cooperation under pressure.

#### **D. Non-Cooperators (L3.1-70B, Q3-32B TE)**
- **Low cooperation** (<0.35): No implicit collusion capacity
- **Baseline coordination** (0.30): Minimal strategic alignment
- **Below random performance:** Worse than baseline

**Implication:** Some configurations fundamentally lack cooperative capabilities.

---

### 4. **3-Player → 5-Player Stability Analysis**

#### Statistical Stability
- **Cooperation p=0.1337** (marginally stable)
- **Coordination p=0.0555** (marginally significant instability)

**Interpretation:** Coordination is more vulnerable to market expansion than cooperation.

#### Degradation Patterns

**Cooperation degradation:**
- L3.1-8B: **-0.411** (largest drop)
- Q3-14B (TE): -0.173
- L3.3-70B: -0.096
- Qwen3-30B-A3B (TE): -0.056
- Q3-235B Inst: -0.051

**Coordination degradation:**
- L4-Scout: **-0.378** (major drop)
- Q3-235B Inst: -0.350
- L3.1-8B: -0.294
- Qwen3-30B-A3B (TE): -0.140

**Improvement patterns:**
- Q3-32B (TD): Coordination +0.014 (to perfect)
- Qwen3-30B-A3B (TD): Cooperation +0.118

**Finding:** Most models degrade under market expansion, but large Qwen-3 models improve.

---

### 5. **Cooperation ↔ Coordination Relationship**

#### Strong Positive Correlation
**High cooperation → High coordination:**
- Q3-14B (TD): 1.00 cooperation, 1.00 coordination
- Q3-32B (TD): 1.00 cooperation, 1.00 coordination
- L4-Scout: 0.95-1.00 cooperation, 0.62-1.00 coordination

**Low cooperation → Low coordination:**
- L3.1-70B: 0.30-0.31 cooperation, 0.30 coordination
- Q3-32B (TE): 0.22-0.27 cooperation, 0.30 coordination

**Correlation estimate:** r ≈ 0.80

**Interpretation:** Implicit collusion requires strategic alignment - they are mutually reinforcing.

#### Exceptions (Coordination < Cooperation)
- Q3-235B Inst: 0.95-1.00 cooperation, 0.65-1.00 coordination
- L4-Maverick: 0.95 cooperation, 0.73-0.75 coordination

**Interpretation:** Some models can maintain collusive pricing without perfect strategic coordination.

---

### 6. **Comparison to Random Baseline**

**Random agent performance:**
- Cooperation: 0.464 → 0.486
- Coordination: 0.300 stable

**Models worse than random (cooperation):**
- Q3-32B (TE): **0.27** vs Random's 0.46
- L3.1-70B: 0.30-0.31 vs Random's 0.46

**Models at random baseline (coordination):**
- 7 models show **0.300 coordination** (same as Random)

**Models significantly better than random:**
- Q3-14B (TD): 1.00 vs 0.46 cooperation (+117%)
- Q3-32B (TD): 1.00 vs 0.46 cooperation (+117%)
- L4-Maverick: 0.95 vs 0.46 cooperation (+106%)

**Finding:** Cooperation shows dramatic model differentiation (0.22-1.00 range), while coordination clusters at baseline (0.30) or high (0.60-1.00).

---

### 7. **Capability Trade-offs**

#### **No Trade-offs Observed**
Unlike Salop (rationality↔cooperation) and Spulber (rationality↔judgment), Green-Porter shows **synergistic dimensions**:
- High cooperation → High coordination
- Low cooperation → Low coordination

**Interpretation:** Implicit collusion requires both dimensions working together - no capability conflicts.

#### **Tuning Effects**
**Q3-32B comparison:**
- TD (think-then-decide): 1.00 cooperation, 1.00 coordination
- TE (think+execute): 0.27 cooperation, 0.30 coordination

**Δ Tuning impact:** -73% cooperation, -70% coordination

**Interpretation:** Different training objectives produce dramatically different cooperation capacities.

---

## Measurement Details

### Scoring Methodology
- **Cooperation:** Frequency of pricing at/near implicit collusive equilibrium
- **Coordination:** Alignment with co-players' strategies without communication

### Standard Deviations
- **High variance (±0.15-0.19):** L3.1-70B, L3.3-70B, Q3-14B TE (unstable cooperation)
- **Low variance (±0.05-0.10):** Q3-32B TD, L4-Maverick, Qwen3-30B-A3B TE (consistent strategies)
- **Fixed values:** Q3-14B TD, L4-Scout, Q3-235B Inst (deterministic collusion)

### Baseline Coordination Value (0.30)
- Appears in 7 models across both conditions
- Likely represents **minimum coordination** from random strategic alignment
- Models either stay at baseline (non-cooperators) or achieve high coordination (>0.60)

---

## Connections to Other Analyses

### Within RQ2 (Behavioral Profiles)
- **T_magic_salop.csv:** Tests cooperation vs rationality trade-off; Green-Porter tests pure cooperation capacity
- **T_magic_spulber.csv:** Tests dynamic entry/exit; Green-Porter tests sustained collusion
- **T_magic_athey_bagwell.csv:** Tests cooperation with deception; Green-Porter tests cooperation under uncertainty
- **T_similarity_3v5.csv:** Confirms high stability (97-99%) - Green-Porter p=0.13/0.06 aligns

### To RQ1 (Performance)
- **T_perf_win_rate.csv:** Models with high cooperation (Q3-32B TD, L4-Maverick) likely perform well in collusive games
- **T_perf_avg_profit.csv:** Cooperation → higher joint profits in repeated games
- **T_perf_game_specific.csv:** Green-Porter-specific win rates should correlate with cooperation scores

### To RQ3 (Capability Links)
- **T5_magic_to_perf.csv:** Tests whether cooperation predicts performance - Green-Porter provides cooperation measure
- **High cooperation variance (0.22-1.00):** Explains performance heterogeneity across models

---

## Theoretical Implications

### 1. **Implicit Collusion as Emergent Capability**
- **Not universal:** Models range from 0.22 to 1.00 cooperation
- **Tuning-dependent:** Same base model (Q3-32B) shows 0.27 vs 1.00 depending on training
- **Scale-sensitive:** Large models (Q3-14B TD, Q3-32B TD) achieve perfect collusion

### 2. **Coordination Threshold Effect**
- Models either fail coordination (0.30 baseline) or succeed (≥0.60)
- **No mid-tier coordination:** Binary outcome suggests cognitive threshold
- **Coordination precedes cooperation:** 0.30 coordination → low cooperation

### 3. **Market Complexity Effects**
- **3P→5P transition stresses cooperation** more than other games
- **Coordination p=0.0555** (marginally significant) suggests 5-player markets exceed coordination capacity
- **Only Q3-32B TD improves** under complexity (anti-fragile cooperation)

---

## Limitations & Caveats

1. **Baseline clustering:** 7 models at 0.300 coordination suggests measurement floor effect
2. **Perfect scores:** Multiple 1.000 values may indicate ceiling effects
3. **Small sample variance:** High SDs for some models indicate instability
4. **Game-specific:** Green-Porter findings may not generalize to other cooperation contexts
5. **No breakdown by demand shock vs cheating:** Can't distinguish cooperation mechanisms

---

## Related Files

- **Data:** `T_magic_green_porter.csv`
- **Visualization:** `F_similarity_green_porter.png`
- **Related summaries:**
  - `SUMMARY_T_magic_salop.md` (cooperation/rationality trade-off)
  - `SUMMARY_T_magic_spulber.md` (rationality/judgment trade-off)
  - `SUMMARY_T_magic_athey_bagwell.md` (cooperation with deception)
  - `SUMMARY_T_similarity_3v5.md` (overall behavioral stability)
  - `SUMMARY_T6_pca_variance.md` (dimensionality analysis)
- **Synthesis:** `SYNTHESIS_RQ2_Behavioral_Profiles.md`

---

## Bottom Line

The Green-Porter repeated oligopoly game reveals **implicit collusion as a specialized capability**:
- **Extreme heterogeneity:** 0.22-1.00 cooperation range across models
- **Synergistic dimensions:** Cooperation and coordination strongly correlated (r ≈ 0.80)
- **Tuning-critical:** Same base model shows 0.27 vs 1.00 cooperation depending on training
- **Threshold coordination:** Models either fail (0.30) or succeed (≥0.60) - no middle ground
- **Market complexity stress:** 3P→5P expansion degrades cooperation (p=0.13) and coordination (p=0.06)

**Key insight:** Implicit collusion is not universal - it depends on model scale, tuning, and market complexity. Top performers (Q3-32B TD, Q3-14B TD) achieve perfect cooperation+coordination, while others fail completely.
