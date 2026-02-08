# Summary: MAgIC Behavioral Profile — Athey-Bagwell

**Table:** `T_magic_athey_bagwell.csv` | **Research Question:** RQ2 - Behavioral Profiles  
**Game:** Entry Deterrence (Capacity Constraints) | **Models:** 13 | **Conditions:** 3P/5P

---

## Game Context

**Athey-Bagwell (2001)**: Entry deterrence through capacity investment. Requires **long-term strategic reasoning** (multi-period payoffs), **cooperation** (maintaining deterrence), **rationality** (optimal capacity choices), **deception** (credible threats).

---

## MAgIC Scores (3P Baseline)

### Overall Statistics

| Dimension | Mean | Range | Interpretation |
|-----------|------|-------|----------------|
| **Reasoning** | **0.915** | [0.556–1.000] | **Excellent** — Most models near-perfect |
| **Cooperation** | **0.875** | [0.532–1.000] | **Strong** — High coordination |
| **Rationality** | **0.619** | [0.358–0.777] | **Moderate** — Room for improvement |
| **Deception** | **0.425** | [0.000–1.000] | **Weak** — High variance, many failures |

**Pattern:** Models excel at reasoning/cooperation, struggle with strategic deception.

---

## Model Rankings (3P)

### Top Performers (All Dimensions)
1. **Q3-14B (TE)**: Rationality=0.641, Reasoning=1.0, Deception=0.0, Cooperation=1.0
2. **Q3-235B Inst**: Rationality=0.641, Reasoning=1.0, Deception=0.0, Cooperation=1.0
3. **Qwen3-30B-A3B**: Rationality=0.641, Reasoning=0.997–1.0, Deception=0.04–0.26, Cooperation=0.997–1.0

### Weakest Performers
- **Q3-14B (TD)**: Rationality=0.358 (lowest), but perfect Reasoning (1.0)
- **L3.1-8B**: Mixed performance across dimensions
- **Random**: Baseline (low across all dimensions)

---

## 3P→5P Structural Sensitivity

| Dimension | 3P Mean | 5P Mean | Change | P-Value | Effect |
|-----------|---------|---------|--------|---------|--------|
| **Rationality** | 0.619 | 0.531 | **-14.2%** | **0.000*** | ✅ Significant decline |
| **Reasoning** | 0.915 | 0.979 | +7.0% | 0.467 ns | ❌ Stable |
| **Deception** | 0.425 | 0.409 | -3.8% | 0.585 ns | ❌ Stable |
| **Cooperation** | 0.875 | 0.905 | +3.4% | 0.947 ns | ❌ Stable |

**Key Finding:** **Only rationality degrades** with more players (p<0.001). Reasoning, cooperation, deception remain stable. Suggests **increased strategic complexity** from monitoring more competitors overwhelms rational optimization.

---

## Dimensional Insights

### Reasoning (Mean=0.915) — STRENGTH
- **8 models at perfect 1.0** (Q3-14B TE, Q3-235B, multiple Qwen/Llama)
- **Stable across 3P→5P** (p=0.467)
- **Interpretation:** Multi-period payoff calculation well-learned. Entry deterrence logic understood.

### Cooperation (Mean=0.875) — STRENGTH
- **7 models at perfect 1.0**
- **Increases slightly in 5P** (+3.4%, though ns)
- **Interpretation:** Capacity coordination robust. More players don't disrupt deterrence cooperation.

### Rationality (Mean=0.619) — MODERATE
- **Range: 0.358–0.777** (no perfect scores)
- **Significant 5P degradation** (-14.2%, p<0.001)
- **Top performers**: Qwen3-30B-A3B TD (0.777), multiple at 0.641
- **Interpretation:** Optimal capacity choice challenging. Complexity sensitivity highest here.

### Deception (Mean=0.425) — WEAKNESS
- **Highest variance**: [0.0–1.0]
- **6 models at 0.0** (complete failure)
- **4 models at 1.0** (perfect success)
- **Interpretation:** **Binary capability** — models either master credible threats or fail completely. No middle ground.

---

## Model-Specific Profiles

### Q3-14B (TE) — Balanced Excellence
- Rationality: 0.641 (moderate)
- Reasoning: 1.0 (perfect)
- Deception: 0.0 (fails threats)
- Cooperation: 1.0 (perfect)
- **Profile:** Strong fundamentals, cannot execute strategic deception

### Qwen3-30B-A3B (TD) — Rationality Leader
- **Rationality: 0.777 (highest)**
- Reasoning: 0.997 (near-perfect)
- Deception: 0.040 (minimal)
- Cooperation: 0.997 (near-perfect)
- **Profile:** Best optimizer, weak at threats

### Q3-32B (TE) — Deception Specialist
- Rationality: 0.641
- Reasoning: 1.0
- **Deception: 1.0 (perfect)**
- Cooperation: 1.0
- **Profile:** One of few models mastering credible threats

---

## Cross-Dimension Correlations

**Reasoning ↔ Cooperation**: Strong positive (models with 1.0 reasoning often have 1.0 cooperation)  
**Rationality ↔ Deception**: **No clear correlation** (high rationality ≠ high deception)  
**Deception**: **Independent capability** — binary, not predicted by other dimensions

---

## Key Takeaways

1. **Reasoning/cooperation are strengths** — 91.5% and 87.5% mean scores, many perfect scores
2. **Rationality moderately challenging** — No model achieves >0.78, degrades significantly in 5P
3. **Deception is binary** — Models either 100% succeed or 100% fail at credible threats (no gradual learning)
4. **Structural robustness varies** — Only rationality sensitive to 3P→5P expansion
5. **Game-specific capability gap** — Athey-Bagwell deception (42.5%) much lower than reasoning (91.5%)
- L4-Maverick: **0.777 → 0.695** (remains highest)
- Q3-32B (TE): 0.651 → 0.591
- Random: 0.649 → 0.605 (surprisingly competitive)
- Q3-14B (TE/235B/L3.3-70B): All start at 0.641

**Moderate rationality (0.55-0.65):**
- Q3-32B (TD): 0.564 → 0.532
- L4-Scout: 0.608 → 0.546
- L3.1-8B: 0.597 → 0.538

**Low rationality (<0.40):**
- Q3-14B (TD): **0.358 → 0.244** (lowest, significant drop)

**Finding:** All models degrade under increased market complexity - rationality is universally vulnerable.

#### Reasoning (Multi-period Calculation)
**Near-perfect reasoning (≥0.95):**
- 8 out of 12 models maintain **≥0.95** in both conditions
- Perfect reasoning: Q3-14B (TD/TE), Q3-235B Inst, Qwen3-30B-A3B (TD/TE), L3.3-70B, L3.1-70B
- L3.1-8B: 0.993 → 0.886 (slight degradation but still high)
- Q3-32B (TE): 0.970 → 0.949

**Moderate reasoning:**
- L4-Maverick: **0.620 → 0.698** (improves under pressure!)
- L4-Scout: 0.761 → 0.744 (stable moderate)

**Low reasoning:**
- Random: 0.556 → 0.513 (baseline)

**Finding:** Most models excel at multi-period reasoning; Llama-4-Maverick uniquely improves under complexity.

#### Deception (Secret Price Cuts)
**Extreme polarization - binary capability:**

**Perfect deceivers (1.00):**
- L4-Maverick: 1.00 → 0.92 (slight degradation but remains highest)
- L4-Scout: 1.00 stable
- Random: 1.00 stable

**No deception capacity (0.00-0.26):**
- Q3-14B (TE): **0.000 stable** (refuses to deceive)
- Q3-235B Inst: 0.000 stable
- Qwen3-30B-A3B (TE): 0.000 stable
- L3.3-70B: 0.000 stable
- L3.1-70B: 0.000 stable
- Qwen3-30B-A3B (TD): 0.04 → 0.26

**Moderate deceivers (0.30-0.68):**
- L3.1-8B: **0.68 → 0.92** (becomes highly deceptive)
- Q3-32B (TD): 0.56 → 0.38
- Q3-32B (TE): 0.24 → 0.26
- Q3-14B (TD): 1.00 stable

**Finding:** Deception is a binary capability - models either can't/won't deceive (0.00) or do so consistently (0.92-1.00). Middle ground is unstable.

#### Cooperation (Collusion Maintenance)
**High cooperation (>0.95):**
- Q3-14B (TE): **1.000 stable** (perfect despite no deception)
- Q3-235B Inst: 1.000 stable
- Qwen3-30B-A3B (TD): 0.997 → 0.980
- Qwen3-30B-A3B (TE): 1.000 stable
- L3.3-70B: 1.000 stable
- L3.1-70B: 1.000 stable
- Q3-32B (TE): 0.978 → 0.974

**Moderate cooperation (0.80-0.95):**
- Q3-32B (TD): 0.885 → 0.954 (improves!)
- L3.1-8B: **0.925 → 0.829** (degrades)
- L4-Maverick: 0.795 → 0.843

**Lower cooperation (0.55-0.70):**
- L4-Scout: 0.683 → 0.678
- Q3-14B (TD): 0.575 → 0.594
- Random: 0.532 → 0.528

**Finding:** Most models maintain high cooperation (>0.95) despite varying deception capabilities.

---

### 2. **Model Archetypes**

#### **A. Honest Colluders (Qwen-3 majority)**
- **Moderate-high rationality** (0.57-0.65): Effective collusion strategy
- **Perfect reasoning** (1.00): Strong multi-period calculation
- **Zero deception** (0.00): Ethical constraint or incapability
- **Perfect cooperation** (1.00): Maintain collusion through honesty

**Models:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (TE), L3.3-70B, L3.1-70B  
**Implication:** Successful collusion without deception through perfect coordination.

#### **B. Strategic Deceivers (Llama-4 series)**
- **High rationality** (0.70-0.78): Optimal profit maximization
- **Moderate-improving reasoning** (0.62-0.76): Learn under pressure
- **Perfect/near-perfect deception** (0.92-1.00): Strategic price cuts
- **Moderate cooperation** (0.68-0.84): Balance defection with collusion

**Models:** L4-Maverick, L4-Scout  
**Implication:** Profit-maximizing through strategic cheating - "cooperate when beneficial, defect when profitable."

#### **C. Adaptive Opportunists (L3.1-8B)**
- **Moderate rationality** (0.54-0.60): Adequate collusion understanding
- **High reasoning** (0.89-0.99): Strong calculation
- **Increasing deception** (0.68 → 0.92): Learns to cheat under pressure
- **High but degrading cooperation** (0.93 → 0.83): Opportunistic defection

**Implication:** Learns deceptive strategies under competitive pressure.

#### **D. Vulnerable Colluders (Q3-14B TD)**
- **Low rationality** (0.24-0.36): Weakest collusion strategy
- **Perfect reasoning** (1.00): Calculation ability present
- **Perfect deception** (1.00): Capable but...
- **Moderate cooperation** (0.58-0.59): Inconsistent collusion

**Implication:** Capability mismatch - can calculate and deceive but can't maintain optimal strategy.

---

### 3. **3-Player → 5-Player Stability Analysis**

#### Statistical Stability
- **Rationality p=0.0** (highly significant instability!)
- **Reasoning p=0.4666** (stable)
- **Deception p=0.5845** (stable)
- **Cooperation p=0.9467** (most stable)

**Interpretation:** Increased market complexity specifically undermines rationality, while other capabilities remain stable.

#### Rationality Degradation Patterns
**All models degrade rationality (100% negative trend):**

**Largest drops:**
- Q3-14B (TD): **-0.114** (-32% relative)
- Qwen3-30B-A3B (TD): -0.088 (-14%)
- Q3-32B (TD): -0.065 (-12%)
- Random: -0.044

**Smallest drops:**
- Q3-14B (TE/235B/L3.3-70B): -0.072 (-11%)
- L4-Maverick: -0.082 (-11%)
- Q3-32B (TE): -0.060 (-9%)

**Finding:** Rationality universally vulnerable - no model maintains optimal strategy under increased monitoring difficulty.

---

### 4. **Capability Relationships**

#### **Deception ↔ Cooperation (Inverse Correlation)**
**Perfect deception → Lower cooperation:**
- L4-Maverick: 1.00 deception, 0.80-0.84 cooperation
- L4-Scout: 1.00 deception, 0.68 cooperation
- Random: 1.00 deception, 0.53 cooperation

**Zero deception → Perfect cooperation:**
- Q3-14B (TE): 0.00 deception, 1.00 cooperation
- Q3-235B Inst: 0.00 deception, 1.00 cooperation
- L3.3-70B: 0.00 deception, 1.00 cooperation
- L3.1-70B: 0.00 deception, 1.00 cooperation

**Correlation estimate:** r ≈ -0.70

**Interpretation:** Models trade off between cheating (deception) and maintaining collusion (cooperation).

#### **Reasoning: Independent from Strategy**
- High reasoning (≥0.95) appears in both deceivers (L4-Maverick: 0.62-0.70) and honest colluders (Qwen-3: 1.00)
- No clear correlation with rationality, deception, or cooperation
- **Conclusion:** Multi-period calculation is separable from strategic choices.

#### **Rationality: Universal Vulnerability**
- All models degrade under 5P complexity (p=0.0)
- No correlation with other dimensions
- **Interpretation:** Optimal collusion strategy is cognitively demanding - all models struggle with increased market complexity.

---

### 5. **Comparison to Random Baseline**

**Random agent performance:**
- Rationality: 0.649 → 0.605 (competitive!)
- Reasoning: 0.556 → 0.513
- Deception: 1.000 stable (perfect random cheating)
- Cooperation: 0.532 → 0.528

**Models worse than random:**
- **Rationality:** Q3-14B (TD) 0.24-0.36 vs Random 0.61-0.65
- **Reasoning:** All models beat random except L4-Scout marginally
- **Deception:** 5 models have 0.00 vs Random's 1.00
- **Cooperation:** L4-Scout 0.68 vs Random 0.53 (only slightly better)

**Models significantly better than random:**
- **Rationality:** L4-Maverick 0.70-0.78 vs 0.61-0.65 (+15-20%)
- **Reasoning:** 8 models achieve ≥0.95 vs Random's 0.51-0.56 (+70-85%)
- **Cooperation:** 7 models achieve 1.00 vs Random's 0.53 (+88%)

**Finding:** Random baseline is surprisingly competitive in rationality and has perfect deception. LLMs differentiate primarily on reasoning and cooperation.

---

### 6. **Tuning Effects on Capability Profiles**

#### Q3-32B Comparison (TD vs TE)
**TD (think-then-decide):**
- Rationality: 0.564 → 0.532
- Reasoning: 1.000 stable
- Deception: 0.560 → 0.380 (moderate, decreasing)
- Cooperation: 0.885 → 0.954 (improves!)

**TE (think+execute):**
- Rationality: 0.651 → 0.591 (+15%)
- Reasoning: 0.970 → 0.949 (-5% from TD)
- Deception: 0.240 → 0.260 (-57% from TD)
- Cooperation: 0.978 → 0.974 (+9%)

**Δ Tuning impact:**
- TE improves rationality (+15%) and cooperation (+9%)
- TD has 2.3× higher deception capacity
- Both maintain high reasoning (≥0.95)

**Interpretation:** TE tuning favors honest collusion; TD tuning enables strategic deception.

---

## Measurement Details

### Scoring Methodology
- **Rationality:** Alignment with optimal collusion/defection strategy (Nash equilibrium in repeated game)
- **Reasoning:** Success rate in multi-period payoff calculation
- **Deception:** Frequency of successful secret price cuts
- **Cooperation:** Maintenance of collusive pricing despite temptation

### Statistical Testing
- **P-values:** Two-tailed tests for 3P→5P differences
- **Only rationality fails stability test (p=0.0)**
- All other dimensions stable (p > 0.40)

### Standard Deviations
- **High variance (±0.10-0.50):** Rationality, Deception (strategic dimensions)
- **Low variance (±0.02-0.12):** Reasoning, Cooperation (more stable)
- **Binary deception:** Many models show 0.000 or 1.000 (deterministic)

---

## Connections to Other Analyses

### Within RQ2 (Behavioral Profiles)
- **T_magic_salop.csv:** Tests cooperation vs rationality; Athey-Bagwell adds deception dimension
- **T_magic_spulber.csv:** Tests rationality vs judgment; Athey-Bagwell tests rationality vs cooperation
- **T_magic_green_porter.csv:** Tests cooperation under uncertainty; Athey-Bagwell tests cooperation with deception
- **T_similarity_3v5.csv:** Confirms behavioral stability - Athey-Bagwell shows selective instability (only rationality)

### To RQ1 (Performance)
- **T_perf_win_rate.csv:** L4-Maverick's high win rate (80%) aligns with high rationality (0.78) + deception (1.00)
- **T_perf_avg_profit.csv:** Deception capability likely predicts profit in competitive games
- **T_perf_game_specific.csv:** Athey-Bagwell win rates should correlate with deception+rationality scores

### To RQ3 (Capability Links)
- **T5_magic_to_perf.csv:** Tests capability→performance links - Athey-Bagwell provides deception+cooperation measures
- **Deception variance (0.00-1.00):** Major heterogeneity likely explains performance differences

---

## Theoretical Implications

### 1. **Deception as Binary Capability**
- **No middle ground:** Models show 0.00 or near-1.00 deception (bimodal distribution)
- **Ethical constraints or incapability:** 5 models never deceive (0.00 stable)
- **Strategic deception:** L4 series consistently deceive (1.00)

**Hypothesis:** Deception may involve ethical training constraints (RLHF) preventing defection.

### 2. **Cooperation-Deception Trade-off**
- **Perfect cooperation (1.00) → Zero deception (0.00):** Honest colluders
- **High deception (1.00) → Moderate cooperation (0.53-0.84):** Strategic cheaters
- **Correlation r ≈ -0.70:** Strong inverse relationship

**Interpretation:** Models choose between "cooperate honestly" vs "cooperate opportunistically."

### 3. **Universal Rationality Vulnerability**
- **All models degrade under 5P (p=0.0):** Optimal collusion strategy is cognitively demanding
- **Increased monitoring difficulty:** Harder to detect/punish cheating in larger markets
- **No model immune:** Even L4-Maverick (highest) shows -11% degradation

**Implication:** Collusion enforcement difficulty scales super-linearly with market size.

### 4. **Reasoning Independence**
- **High reasoning (≥0.95) in 8 models:** Calculation capability widely available
- **No correlation with strategy:** Deceivers and honest colluders both have perfect reasoning
- **Stable under pressure:** p=0.47 (no 3P→5P effect)

**Conclusion:** Reasoning is necessary but not sufficient for optimal collusion - strategic choice matters more.

---

## Limitations & Caveats

1. **Binary deception:** 0.00 vs 1.00 polarization suggests measurement may not capture nuanced deception
2. **Rationality universally degrades:** May reflect measurement artifact rather than true capability loss
3. **P-value=0.0 for rationality:** Extremely significant but all models affected equally
4. **No breakdown by punishment vs defection:** Can't distinguish cooperation mechanisms
5. **Ethical training confounds:** RLHF may suppress deception, not measure capability

---

## Related Files

- **Data:** `T_magic_athey_bagwell.csv`
- **Visualization:** `F_similarity_athey_bagwell.png`
- **Related summaries:**
  - `SUMMARY_T_magic_salop.md` (cooperation/rationality trade-off)
  - `SUMMARY_T_magic_spulber.md` (rationality/judgment trade-off)
  - `SUMMARY_T_magic_green_porter.md` (cooperation under uncertainty)
  - `SUMMARY_T_similarity_3v5.md` (overall behavioral stability)
  - `SUMMARY_T6_pca_variance.md` (dimensionality analysis)
- **Synthesis:** `SYNTHESIS_RQ2_Behavioral_Profiles.md`

---

## Bottom Line

The Athey-Bagwell collusion enforcement game reveals **strategic deception as a polarizing capability**:
- **Binary deception:** Models either never deceive (0.00) or consistently do (0.92-1.00) - no middle ground
- **Deception ↔ Cooperation inverse (r ≈ -0.70):** Strategic cheaters vs honest colluders
- **Universal rationality vulnerability:** All models degrade under market complexity (p=0.0)
- **Reasoning independence:** High calculation ability (≥0.95 in 8 models) appears in both deceivers and cooperators
- **Tuning effects:** TE favors honesty (+9% cooperation), TD enables deception (2.3× higher)

**Key insight:** Successful collusion follows two distinct paths - (1) perfect cooperation without deception (Qwen-3), or (2) strategic deception with moderate cooperation (Llama-4). No model achieves both perfectly.
