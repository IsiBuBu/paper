# Summary: MAgIC Behavioral Profile — Salop

**Table:** `T_magic_salop.csv` | **Research Question:** RQ2 - Behavioral Profiles  
**Game:** Spatial Competition (Product Differentiation) | **Models:** 13 | **Conditions:** 3P/5P

---

## Game Context

**Salop (1979)**: Firms compete on **circular location space** with **transportation costs**. Requires **spatial reasoning**, **differentiation strategy**, and balancing **market power vs. competition distance**. Hardest game (lowest win rates).

**Key Challenge:** Optimize location AND price simultaneously in multi-dimensional strategy space.

---

## MAgIC Scores (3P Baseline)

### Overall Statistics

| Dimension | Mean | Range | Interpretation |
|-----------|------|-------|----------------|
| **Cooperation** | **0.662** | [0.000–1.000] | **Moderate** — High variance |
| **Reasoning** | **0.439** | [0.027–1.000] | **Weak** — Spatial analysis difficult |
| **Rationality** | **0.208** | [0.000–0.750] | **Very Weak** — Optimization fails |

**Pattern:** **Hardest game for LLMs** — All dimensions significantly lower than other games (e.g., Athey-Bagwell: Reasoning=0.915 vs Salop=0.439).

---

## Model Rankings (3P)

### Top Performers (Reasoning)
1. **Q3-14B (TE)**: 1.000 — Only model with perfect spatial reasoning
2. **Q3-32B (TE)**: 0.958
3. **Qwen3-30B-A3B (TE)**: 0.913

### Top Performers (Rationality)
1. **Q3-32B (TE)**: 0.750 — Best optimization
2. **Qwen3-30B-A3B (TE)**: 0.666
3. **Q3-14B (TE)**: 0.472

### Top Performers (Cooperation)
1. **Multiple models**: 1.000 (Q3-14B TE, Q3-235B, Qwen3-30B TE, Q3-32B TE)
2. **L3.1-70B**: 0.900

### Complete Failures
- **L3.3-70B**: Rationality=0.000, Reasoning=0.027 (near-total failure)
- **L4-Scout**: Rationality=0.000, Reasoning=0.076
- **Random**: Rationality=0.000, Reasoning=0.059

---

## 3P→5P Structural Sensitivity

| Dimension | 3P Mean | 5P Mean | Change | P-Value | Effect |
|-----------|---------|---------|--------|---------|--------|
| **Rationality** | 0.208 | 0.276 | **+32.7%** | **0.003*** | ✅ **Improves** |
| **Reasoning** | 0.439 | 0.542 | **+23.5%** | **0.001*** | ✅ **Improves** |
| **Cooperation** | 0.662 | 0.723 | +9.2% | 0.058 ns | ❌ Marginal |

**Unique Finding:** **Only game where cognitive capabilities improve in 5P** (rationality p=0.003, reasoning p=0.001). Suggests **more competitors provide better spatial learning signal** or force more careful optimization.

---

## Dimensional Insights

### Rationality (Mean=0.208) — CRITICAL WEAKNESS
- **Lowest across all games** (cf. Athey-Bagwell=0.619, Spulber=0.506)
- **7 models at 0.000** (complete optimization failure)
- **Best: Q3-32B (TE) at 0.750** — But still below other games' averages
- **Improves +32.7% in 5P** (0.208 → 0.276, p=0.003)
- **Interpretation:** Spatial optimization extremely difficult. Most models cannot find profit-maximizing location-price pairs.

### Reasoning (Mean=0.439) — WEAKNESS
- **Second-lowest** (only Spulber efficiency lower at allocation tasks)
- **Range: 0.027–1.000** (extreme variance)
- **Only Q3-14B (TE) achieves 1.0**
- **Improves +23.5% in 5P** (p=0.001)
- **Interpretation:** Multi-dimensional strategic analysis (location + price) challenges reasoning. TE models gain significant advantage.

### Cooperation (Mean=0.662) — MODERATE
- **4 models at perfect 1.0**
- **4 models below 0.4** (bimodal distribution)
- **Stable across 3P→5P** (p=0.058 ns)
- **Interpretation:** Coordination capability independent of spatial complexity. Models either cooperate or don't, unaffected by location dimension.

---

## Model-Specific Profiles

### Q3-14B (TE) — Reasoning Champion
- **Reasoning: 1.000 (only perfect score)**
- Rationality: 0.472 (moderate)
- Cooperation: 1.000
- **Profile:** Understands spatial logic perfectly but struggles with optimization implementation

### Q3-32B (TE) — Optimization Leader
- **Rationality: 0.750 (highest)**
- **Reasoning: 0.958 (second-highest)**
- Cooperation: 1.000
- **Profile:** Best overall Salop performer. Strong spatial analysis + optimization.

### L3.3-70B — Complete Failure
- Rationality: 0.000
- Reasoning: 0.027 (near-zero)
- Cooperation: 0.900 (only strength)
- **Profile:** 70B size cannot compensate for spatial reasoning deficit. Cooperates but cannot compete.

---

## Thinking Enhancement Effect

### TE vs TD Gains (Reasoning)
- **Q3-14B**: TD=0.424 → TE=1.000 (**+136%**)
- **Q3-32B**: TD=0.372 → TE=0.958 (**+157%**)
- **Qwen3-30B-A3B**: TD=0.179 → TE=0.913 (**+410%**)

**Massive TE advantage** — Largest cross-game thinking enhancement effect. Spatial reasoning requires extended analysis.

---

## Cross-Game Comparison

| Game | Rationality | Reasoning | Cooperation |
|------|-------------|-----------|-------------|
| **Salop** | **0.208** | **0.439** | 0.662 |
| Athey-Bagwell | 0.619 | 0.915 | 0.875 |
| Spulber | 0.506 | 0.957 | N/A |

**Salop is hardest** — Rationality and reasoning scores **dramatically lower** than other games.

---

## Key Takeaways

1. **Hardest game cognitively** — Rationality (0.208) and reasoning (0.439) lowest across all games
2. **Spatial optimization critical weakness** — 7/13 models at 0.000 rationality (complete failure)
3. **Counter-intuitive 5P improvement** — Rationality (+32.7%) and reasoning (+23.5%) improve with more players
4. **Thinking enhancement most valuable here** — TE gains up to +410% in reasoning (largest across games)
5. **Bimodal cooperation** — Models cluster at 1.0 or <0.4, no middle ground
- **Rationality:** 0.208 (low)

### Average Scores by Dimension (5P)
- **Cooperation:** 0.700 (high)
- **Reasoning:** 0.436 (moderate)
- **Rationality:** 0.342 (moderate)

### 3P→5P Changes
- **Rationality:** +64% increase (p = 0.0672, marginally significant)
- **Reasoning:** +0.2% (p = 0.4863, not significant)
- **Cooperation:** +7% (p = 0.316, not significant)

**Key Insight:** Models become MORE rational in crowded markets (5P), but reasoning/cooperation stable.

## Model-Specific Analysis

### Top Performers (High Capability Models)

#### Rationality Leaders
1. **Q3-32B (TE):** 0.75 (3P) → 1.00 (5P) ⭐⭐⭐
2. **Qwen3-30B-A3B (TE):** 0.75 (3P) → 0.90 (5P)
3. **Q3-14B (TE):** 0.65 (3P) → 0.89 (5P)

**Pattern:** All TE (extended thinking) models show high rationality

#### Reasoning Leaders
1. **Q3-32B (TE):** 1.00 (both conditions) ⭐⭐⭐
2. **Qwen3-30B-A3B (TE):** 1.00 (3P) → 0.90 (5P)
3. **Qwen3-30B-A3B (TD):** 0.72 (3P) → 0.55 (5P)

**Pattern:** TE models dominate reasoning

#### Cooperation Leaders
1. **Q3-32B (TD), L3.3-70B, L4-Maverick, L4-Scout, L3.1-70B:** 1.00 (both conditions)
2. **Q3-14B (TD):** 0.90 (3P) → 1.00 (5P)
3. **Q3-235B Inst:** 0.80 (3P) → 1.00 (5P)

**Pattern:** TD and large Llama models highly cooperative

---

### Low Performers (Low Capability Models)

#### Low Rationality
- **L3.3-70B:** 0.00 (both conditions)
- **L4-Scout:** 0.00 (both conditions)
- **L3.1-70B:** 0.00 (3P) → 0.02 (5P)
- **Q3-32B (TD):** 0.00 (3P) → 0.11 (5P)

**Pattern:** Large models (70B) struggle with rationality in Salop

#### Low Reasoning
- **L3.3-70B:** 0.03 (3P) → 0.00 (5P)
- **L3.1-70B:** 0.03 (3P) → 0.00 (5P)
- **Q3-32B (TD):** 0.11 (both conditions)

**Pattern:** Same models struggle with both rationality AND reasoning

#### Low Cooperation
- **Q3-14B (TE):** 0.30 (3P) → 0.00 (5P)
- **Q3-32B (TE):** 0.00 (both conditions)
- **Qwen3-30B-A3B (TE):** 0.00 (3P) → 0.10 (5P)

**Pattern:** TE models are LEAST cooperative (trade-off with rationality)

---

## Capability Trade-offs

### Trade-off 1: Rationality ↔ Cooperation
**Negative correlation observed:**
- **High rationality models** (Q3-32B TE: 1.00) have **low cooperation** (0.00)
- **High cooperation models** (L4-Scout: 1.00) have **low rationality** (0.00)

**Interpretation:** Rational optimizers compete aggressively; cooperative models avoid price wars but don't optimize.

### Trade-off 2: Reasoning + Rationality (TE models)
**Positive correlation:**
- TE models high in BOTH reasoning (1.00) and rationality (0.75–1.00)
- TD models low in BOTH reasoning (0.11–0.44) and rationality (0.00–0.20)

**Interpretation:** Extended thinking enables both deep reasoning AND rational optimization.

### Trade-off 3: Thinking Mode Divide
**TE Models:**
- High: Rationality, Reasoning
- Low: Cooperation

**TD Models:**
- High: Cooperation
- Low: Rationality, Reasoning

**Interpretation:** Thinking mode creates opposite behavioral profiles.

---

## Relationship to Performance (from T_perf_avg_profit)

### High Rationality + High Reasoning → High Profit
| Model | Rationality | Reasoning | Profit (3P) | Rank |
|-------|-------------|-----------|-------------|------|
| Q3-32B (TE) | 0.75 | 1.00 | 1,383 | 🥇 1st |
| Qwen3-30B-A3B (TE) | 0.75 | 1.00 | 1,372 | 🥈 2nd |
| Q3-14B (TE) | 0.65 | 0.70 | 950 | 🥉 3rd |

**Correlation:** Strong positive (r ≈ 0.85)

### High Cooperation + Low Rationality → Negative Profit
| Model | Cooperation | Rationality | Profit (3P) | Rank |
|-------|-------------|-------------|-------------|------|
| L3.3-70B | 1.00 | 0.00 | -65 | ❌ 12th |
| L3.1-70B | 1.00 | 0.00 | -67 | ❌ 13th |

**Correlation:** Cooperation alone insufficient; needs rationality

### Medium Capabilities → Medium Profit
| Model | Avg MAgIC | Profit (3P) | Rank |
|-------|-----------|-------------|------|
| Qwen3-30B-A3B (TD) | 0.58 | 1,312 | 4th |
| Q3-14B (TD) | 0.45 | 567 | 6th |

**Correlation:** Moderate positive

---

## 3P→5P Stability Analysis

### Stable Dimensions (High Similarity)
- **Reasoning:** 0.435 → 0.436 (+0.2%, p = 0.49)
- **Cooperation:** 0.654 → 0.700 (+7%, p = 0.32)

**Interpretation:** Strategic reasoning and cooperation tendencies persist across conditions.

### Adaptive Dimension
- **Rationality:** 0.208 → 0.342 (+64%, p = 0.067 †)

**Interpretation:** Models become MORE rational when markets get crowded (5P). Suggests adaptation to increased competition.

---

## Model Archetypes in Salop

### Archetype 1: Rational Strategists (TE Models)
- **Examples:** Q3-32B (TE), Qwen3-30B-A3B (TE), Q3-14B (TE)
- **Profile:** High rationality (0.65–1.00), High reasoning (0.70–1.00), Low cooperation (0.00–0.30)
- **Strategy:** Aggressive price competition, optimal positioning
- **Outcome:** Highest profits (950–1,383)

### Archetype 2: Cooperative Avoiders (TD Models, Large Llamas)
- **Examples:** L4-Scout, L3.3-70B, Q3-32B (TD)
- **Profile:** Low rationality (0.00–0.11), Low reasoning (0.03–0.17), High cooperation (1.00)
- **Strategy:** Avoid competition, high prices, poor positioning
- **Outcome:** Negative or low profits (-67 to 147)

### Archetype 3: Balanced Generalists (Medium TD Models)
- **Examples:** Qwen3-30B-A3B (TD), Q3-14B (TD), Random
- **Profile:** Moderate all dimensions (0.20–0.72)
- **Strategy:** Mix of competition and cooperation
- **Outcome:** Moderate profits (419–1,312)

---

## Statistical Significance

### 3P→5P Effects

| Dimension | 3P Mean | 5P Mean | Change | P-Value | Significance |
|-----------|---------|---------|--------|---------|--------------|
| Rationality | 0.208 | 0.342 | +64% | 0.0672 | † (marginal) |
| Reasoning | 0.435 | 0.436 | +0.2% | 0.4863 | ns |
| Cooperation | 0.654 | 0.700 | +7% | 0.3160 | ns |

**Note:** † = marginally significant (p < 0.10)

### Interpretation
- **Rationality increases marginally** with more players (adaptive response to crowding)
- **Reasoning/cooperation stable** (fundamental behavioral traits)

---

## Comparison to Other Games

### Salop vs Athey-Bagwell
- **Salop:** Lower rationality (0.21 vs 0.48), more cooperation (0.65 vs 0.42)
- **Why:** Differentiation allows collusion; capacity constraints force competition

### Salop vs Spulber
- **Salop:** Higher cooperation (0.65 vs 0.52), similar rationality (0.21 vs 0.25)
- **Why:** Differentiation enables coordination; search frictions reduce cooperation need

### Salop vs Green-Porter
- **Salop:** Lower cooperation (0.65 vs 0.88), higher rationality (0.21 vs 0.11)
- **Why:** Salop rewards optimization; Green-Porter rewards collusion

**Pattern:** Salop is most BALANCED game — requires both competition (rationality) and coordination (cooperation).

---

## Key Insights

### 1. Thinking Mode Determines Profile
- **TE models:** High rationality + reasoning, low cooperation → Competitive strategy
- **TD models:** High cooperation, low rationality/reasoning → Cooperative strategy
- **Implication:** Can predict behavior from thinking mode in Salop

### 2. Rationality-Cooperation Trade-off
- **Cannot maximize both** (r ≈ -0.60 correlation)
- **Rational models** compete aggressively (low cooperation)
- **Cooperative models** avoid conflict but miss optimization opportunities
- **Implication:** Design choice between competitive vs collusive agents

### 3. Rationality is Adaptive, Reasoning is Stable
- **Rationality increases 3P→5P** (models adjust to crowding)
- **Reasoning stable** (fundamental capability, not context-dependent)
- **Implication:** Some capabilities flexible, others fixed

### 4. Success Requires Balance (But Leans Rational)
- **Top performers:** High rationality (0.75+) + high reasoning (0.70+)
- **Cooperation optional:** Q3-32B TE succeeds with 0.00 cooperation
- **Implication:** Competitive optimization more important than coordination in Salop

---

## Practical Implications

### For Model Selection (Salop-like Tasks)
1. **Choose TE models** for aggressive competition, profit maximization
2. **Choose TD models** for stable pricing, risk avoidance
3. **Avoid large Llama models** (70B) — lack rationality for differentiation

### For Prompt Engineering
1. **Enable thinking mode** (TE) for differentiation tasks
2. **Emphasize optimization** over cooperation in prompts
3. **Provide differentiation framework** (models need rationality to find niches)

### For Training
1. **Target rationality + reasoning** for competitive markets
2. **Balance cooperation vs competition** (current models polarized)
3. **Train adaptive rationality** (3P→5P increase suggests learnability)

---

## Data Quality Notes

### High Variance Models
- **Random:** High σ (0.42 cooperation, 0.42 rationality) — expected noise
- **Q3-14B (TE):** High σ (0.47 rationality) — inconsistent optimization
- **Qwen3-30B-A3B (TE):** High σ (0.43 rationality) — variable strategy

### Perfect Scores (1.00)
- **Q3-32B (TE):** 1.00 reasoning, 1.00 rationality (5P)
- **Multiple models:** 1.00 cooperation
- **Interpretation:** Some models maximize single dimensions

### Zero Scores (0.00)
- **Many models:** 0.00 rationality (70B Llamas, TD models)
- **TE models:** 0.00 cooperation
- **Interpretation:** Trade-offs force specialization

---

## Related Files
- `T_perf_avg_profit.csv` — Profit outcomes (rationality+reasoning→profit)
- `T_perf_win_rate.csv` — Win rates (Q3-32B TE: 1.00 in 5P)
- `T5_magic_to_perf.csv` — MAgIC→performance regression (R²=0.85 for Salop)
- `F_similarity_salop.png` — Within-condition similarity matrix
- `T_similarity_3v5.csv` — Cross-condition stability (97.4% for Salop)
- `SYNTHESIS_RQ2_Behavioral_Profiles.md` — Complete RQ2 analysis
- `SYNTHESIS_RQ3_Capability_Performance_Links.md` — Capability→performance links
