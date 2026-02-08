# Summary: Behavioral Similarity Heatmap — Athey-Bagwell

**Figure:** `F_similarity_athey_bagwell.png` | **Research Question:** RQ2 - Behavioral Profiles  
**Data:** Cosine similarity matrix (13×13 models) | **Dimensions:** Rationality, Reasoning, Deception, Cooperation

---

## Visualization Overview

**13×13 symmetric heatmap** showing pairwise behavioral similarity based on Athey-Bagwell MAgIC profiles (4 dimensions). **Most complex similarity space** due to deception dimension.

**Key Feature:** Tests whether deception capability (binary: 0.0 vs 1.0) creates distinct clusters.

---

## Clustering Pattern Analysis

### Cluster 1: Perfect Reasoners (Reasoning=1.0, Variable Deception)
**Members:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (both modes), Q3-32B (TE), L3.3-70B, L4-Maverick, L3.1-70B

**Behavioral Profile:**
- Reasoning: 1.000 (perfect multi-period understanding)
- Cooperation: 0.997–1.000 (near-perfect coordination)
- Rationality: 0.641–0.777 (moderate optimization)
- **Deception: BINARY split** (some 0.0, some 1.0)

**Heatmap Pattern:** Large dark red/orange block — **8 models** share perfect reasoning/cooperation

**Interpretation:** Entry deterrence logic well-learned. Deception splits otherwise similar models.

---

### Cluster 2: Deception Masters (Deception=1.0)
**Members:** Q3-32B (TE), Qwen3-30B-A3B (TD), L4-Maverick, L4-Scout

**Distinguishing Feature:** **Credible threat capability** (deception=1.0)

**Sub-clustering:**
- Q3-32B (TE) ↔ Qwen3-30B-A3B (TD): High similarity (~0.85) — Both perfect reasoners + deception
- L4 models: Moderate similarity (~0.75) — Deception but lower cooperation

**Interpretation:** Deception capability **not sufficient** for tight clustering. Cooperation + deception needed.

---

### Cluster 3: Deception Failures (Deception=0.0)
**Members:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (TE), Q3-32B (TD), L3.3-70B, L3.1-70B

**Profile:** Perfect reasoning/cooperation but **cannot execute strategic threats**

**Similarity:** High within-cluster (>0.90) — Shared incapacity creates tight grouping

**Interpretation:** **Binary capability** — Models either master threats or completely fail. No gradual learning.

---

## Within-Model Divergence (TD vs TE)

### Q3-14B: Stable Profiles
- **Similarity: ~0.85** (high)
- **TD**: Rationality=0.358, Reasoning=1.0, Deception=1.0, Cooperation=0.575
- **TE**: Rationality=0.641 (+79%), Reasoning=1.0, Deception=0.0 (-100%), Cooperation=1.0 (+74%)
- **Effect:** TE gains cooperation/rationality but **loses deception** (trade-off)

### Qwen3-30B-A3B: Deception Shift
- **Similarity: ~0.75** (moderate)
- **TD**: Deception=0.040 (minimal), Cooperation=0.997
- **TE**: Deception=0.260 (+550%), Cooperation=0.980
- **Effect:** TE enhances deception moderately but doesn't achieve mastery (1.0)

### Q3-32B: Deception Swap
- **Similarity: ~0.80**
- **TD**: Deception=1.0, Cooperation=1.0
- **TE**: Deception=1.0, Cooperation=1.0
- **Effect:** **Most stable** TD-TE pair — Both master deception + cooperation

**Key Finding:** Athey-Bagwell shows **highest TD-TE stability** (0.75–0.85) compared to Salop (0.60–0.80). Entry deterrence profiles more robust.

---

## Family Clustering Assessment

### Qwen Family Internal Similarity
**Expected:** High (>0.85)

**Observed:**
- Q3-14B (TE) ↔ Q3-235B Inst: **>0.95** (both deception=0, cooperation=1.0)
- Q3-32B (TD) ↔ Qwen3-30B-A3B (TD): ~0.85 (both deception=1.0 or near)
- Cross-tuning pairs: 0.75–0.85 (moderate)

**Conclusion:** **Moderate family clustering** — Strongest among all games, driven by shared reasoning=1.0 baseline

---

### Llama Family Internal Similarity
**Observed:**
- L4-Maverick ↔ L4-Scout: ~0.80 (both deception=1.0)
- L3.3-70B ↔ L3.1-70B: ~0.90 (both deception=0.0, reasoning=1.0)
- Cross-generation: 0.70–0.80

**Conclusion:** **Stronger Llama clustering** in Athey-Bagwell than other games. Deception capability aligns within generations.

---

## Deception-Driven Clustering

### Deception=1.0 Group
**Members:** Q3-32B (TE), Qwen3-30B-A3B (TD), L4-Maverick, L4-Scout

**Similarity:** 0.75–0.85 (orange cells)

**Pattern:** Forms **diagonal sub-block** in heatmap

**Interpretation:** Shared capability creates moderate cohesion, but cooperation levels vary (0.575–1.0)

---

### Deception=0.0 Group
**Members:** Q3-14B (TE), Q3-235B, Qwen3-30B-A3B (TE), Q3-32B (TD), L3.3-70B, L3.1-70B

**Similarity:** **0.90–0.97** (darkest red block in heatmap)

**Pattern:** **Tightest cluster across all games**

**Interpretation:** Shared failure + perfect reasoning/cooperation → **highest homogeneity**. Incapacity more defining than capability.

---

## Outlier Identification

### Q3-14B (TD) — Rationality Outlier
- **Similarity:** 0.60–0.75 to most models (yellow-orange)
- **Profile:** **Rationality=0.358 (lowest)**, Reasoning=1.0, Deception=1.0, Cooperation=0.575
- **Interpretation:** Lowest rationality creates moderate isolation despite deception capability

### Random — Complete Isolation
- **Similarity:** 0.20–0.45 (white)
- **Interpretation:** Baseline, as expected

---

## Quantitative Distribution

| Similarity Range | Pair Count | Interpretation |
|------------------|------------|----------------|
| **>0.90** (dark red) | ~15 pairs | Deception=0 cluster (largest in dataset) |
| **0.80-0.90** (red-orange) | ~25 pairs | Within-family, cross-deception |
| **0.70-0.80** (orange) | ~30 pairs | Cross-family, moderate TD-TE |
| **<0.70** (yellow-white) | ~25 pairs | Q3-14B (TD), Random outliers |

**Pattern:** **Tightest clustering** among all games (most pairs >0.80). Athey-Bagwell profiles most homogeneous.

---

## Cross-Game Comparison

**Athey-Bagwell vs Salop Clustering:**
- **Athey-Bagwell**: 15 pairs >0.90 (tight deception=0 cluster)
- **Salop**: 10 pairs >0.85 (looser TE cooperator cluster)

**Interpretation:** Entry deterrence creates **stronger behavioral convergence** than spatial competition. Perfect reasoning (91.5% mean) creates shared foundation.

---

## Key Takeaways

1. **Deception creates binary split** — Models cluster by deception=0.0 (6 models) vs 1.0 (4 models), not gradual
2. **Tightest clustering across all games** — 15 pairs >0.90, driven by shared reasoning=1.0 + cooperation≈1.0
3. **Deception failure more cohesive than success** — Deception=0 cluster (>0.90 similarity) tighter than deception=1.0 cluster (0.75–0.85)
4. **Highest TD-TE stability** — Athey-Bagwell shows 0.75–0.85 similarity (vs Salop 0.60–0.80)
5. **Family effects strongest here** — Qwen family shows >0.85 internal similarity (vs <0.70 in Salop)
10. L4-Scout
11. L3.1-70B
12. L3.1-8B
13. Random

---

## Key Visual Patterns

### 1. **High-Similarity Clusters (Dark Red Regions)**

#### **Cluster A: Honest Cooperators (Qwen-3 TE + Llama Non-Deceptive)**
**Expected members:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (TE), Q3-32B (TE), L3.3-70B, L3.1-70B

**Behavioral profile:**
- Rationality: 0.57-0.64 (moderate): Balanced profit-seeking
- Reasoning: 0.95-1.00 (near-perfect): Strong calculation
- Deception: 0.00-0.26 (very low): Minimal cheap talk
- Cooperation: 0.97-1.00 (very high): Near-perfect collusion

**Similarity pattern:**
- Very high intra-cluster similarity (>0.90)
- Forms **largest dark red block** (6 models)
- Unified strategic approach: "cooperate honestly without deception"
- **Cross-family clustering:** Includes both Qwen and Llama models

**Interpretation:** Models with zero/minimal deception cluster together regardless of family. **Honesty + cooperation** defines this archetype. Strategic communication is ignored; actions alone maintain collusion.

#### **Cluster B: Strategic Deceivers (Llama Deceptive + Q3-32B TD + L3.1-8B)**
**Expected members:** L4-Maverick, L4-Scout, Q3-32B (TD), L3.1-8B, Random

**Behavioral profile:**
- Rationality: 0.61-0.78 (moderate-high): Profit-seeking with strategy
- Reasoning: 0.62-1.00 (high-perfect): Variable calculation ability
- Deception: 0.38-1.00 (moderate-high): Active cheap talk usage
- Cooperation: 0.68-0.95 (moderate-high): Collusion with strategic signaling

**Similarity pattern:**
- Moderate to high intra-cluster similarity (0.75-0.88)
- Forms orange/red block
- **Heterogeneous group:** Wide deception variance (38-100%)
- Unified by "active communication strategy"

**Interpretation:** Models that use deception cluster together, but **deception level varies widely**. Some use it moderately (Q3-32B TD: 38-56%), others maximally (L4-Maverick, L4-Scout, Random: 92-100%). Cluster defined by **communication engagement**, not cooperation level.

---

### 2. **Within-Model Variance (TD vs TE Divergence)**

#### **Q3-14B: TD vs TE Comparison**
**Expected similarity:** 0.65-0.75 (moderate divergence)

**TD profile:**
- Rationality: 0.244 (low)
- Reasoning: 1.000 (perfect)
- Deception: 1.000 (perfect) - **maximal cheap talk**
- Cooperation: 0.594 (moderate)

**TE profile:**
- Rationality: 0.569 (moderate) - **133% increase**
- Reasoning: 1.000 (perfect) - stable
- Deception: 0.000 (none) - **100% drop**
- Cooperation: 1.000 (perfect) - **69% increase**

**Visual pattern:**
- Orange/yellow cell (moderate-low similarity)
- **25-35% divergence**
- TD → Cluster B (strategic deceiver), TE → Cluster A (honest cooperator)
- **Complete deception reversal:** 1.000 → 0.000

**Interpretation:** Tuning creates **complete behavioral flip** on deception dimension. TD uses maximal cheap talk with moderate cooperation; TE abandons communication entirely with perfect cooperation. **Communication strategy vs action-based collusion trade-off.**

#### **Q3-32B: TD vs TE Comparison**
**Expected similarity:** 0.70-0.80 (moderate divergence)

**TD profile:**
- Rationality: 0.532 (moderate)
- Reasoning: 1.000 (perfect)
- Deception: 0.380 (moderate) - **selective cheap talk**
- Cooperation: 0.954 (very high)

**TE profile:**
- Rationality: 0.591 (moderate) - 11% increase
- Reasoning: 0.949 (near-perfect) - stable
- Deception: 0.260 (low) - 32% drop
- Cooperation: 0.974 (very high) - stable

**Visual pattern:**
- Red/orange cell (high-moderate similarity)
- **20-30% divergence** - smaller than Q3-14B
- TD → Cluster B (moderate deceiver), TE → Cluster A (minimal deceiver)
- **Partial deception reduction:** 0.380 → 0.260

**Interpretation:** Tuning reduces deception but doesn't eliminate it. Both maintain high cooperation (~0.95). TD uses moderate cheap talk; TE uses less. **Both prefer action-based collusion over communication.**

#### **Qwen3-30B-A3B: TD vs TE Comparison**
**Expected similarity:** 0.75-0.85 (moderate divergence)

**TD profile:**
- Rationality: 0.553 (moderate)
- Reasoning: 1.000 (perfect)
- Deception: 0.260 (low) - **minimal cheap talk**
- Cooperation: 0.980 (very high)

**TE profile:**
- Rationality: 0.569 (moderate) - stable
- Reasoning: 1.000 (perfect) - stable
- Deception: 0.000 (none) - 100% drop
- Cooperation: 1.000 (perfect) - stable

**Visual pattern:**
- **Orange/red cell** (moderate-high similarity)
- **15-25% divergence**
- Both near Cluster A (honest cooperators)
- **Deception elimination:** 0.260 → 0.000

**Interpretation:** Tuning eliminates low-level deception, moving model from "minimal communication" to "zero communication." Both prioritize cooperation; TD occasionally uses cheap talk, TE never does.

---

### 3. **Family Clustering Analysis**

#### **Qwen-3 Family (7 models)**
**Expected pattern:** Moderate family clustering, TE models group together

**Observation:**
- **TE models → Cluster A (honest):** Q3-14B, Q3-235B, Qwen3-30B-A3B, Q3-32B
  - All have deception ≤ 0.26
  - All have cooperation ≥ 0.97
- **TD models split:**
  - Q3-14B TD → Cluster B (maximal deceiver)
  - Q3-32B TD → Cluster B (moderate deceiver)
  - Qwen3-30B-A3B TD → Near Cluster A (minimal deceiver)
- **Within-family variance:** 50-85% similarity range

**Interpretation:** Qwen family shows **TE cohesion** (all TE models cluster together) but **TD heterogeneity** (TD models scatter by deception level). Tuning toward engagement **consistently eliminates deception**. Architecture alone predicts little; tuning mode + deception level predicts clustering.

#### **Llama-3/4 Family (5 models)**
**Expected pattern:** Deception-based split

**Observation:**
- **High deception models → Cluster B:**
  - L4-Maverick (deception: 0.92-1.00)
  - L4-Scout (deception: 1.00)
  - Random (deception: 1.00)
- **Zero deception models → Cluster A:**
  - L3.3-70B (deception: 0.00)
  - L3.1-70B (deception: 0.00)
- **Moderate deception model:**
  - L3.1-8B (deception: 0.68-0.92) → Cluster B
- **Within-family variance:** 60-90% similarity range

**Interpretation:** Llama family is **divided by deception capability**. Llama-4 models are maximal deceivers; Llama-3 models (except L3.1-8B) avoid deception entirely. **Version/size predicts deception propensity better than family membership.**

---

### 4. **Outliers and Distinctive Profiles**

#### **Random Baseline: Accidental Deceiver**
**Expected similarity:** 0.40-0.60 (low to moderate)

**Behavioral profile:**
- Rationality: 0.605 (moderate)
- Reasoning: 0.513 (moderate-low)
- Deception: 1.000 (maximal) - **always sends signals**
- Cooperation: 0.528 (moderate)
- **Noisy but deceptive** - signals randomly

**Visual pattern:**
- Orange/yellow cells with most models
- **Higher similarity to Cluster B** (deceptive models)
- **Red cell with L4-Maverick, L4-Scout** (fellow maximal deceivers)

**Interpretation:** Random baseline **accidentally mimics strategic deceivers** due to maximal signaling. In Athey-Bagwell, random communication creates superficial similarity to strategic cheap talk. **Higher LLM-like similarity than in other games** due to deception dimension.

#### **Q3-14B (TD): Maximal Deceiver**
**Expected similarity:** 0.70-0.85 (high with Cluster B)

**Behavioral profile:**
- Rationality: 0.244 (low) - **lowest among Qwen**
- Reasoning: 1.000 (perfect)
- Deception: 1.000 (maximal) - **unique among Qwen**
- Cooperation: 0.594 (moderate) - **lowest among Qwen**

**Visual pattern:**
- Red cells with L4-Maverick, L4-Scout, Random (fellow maximal deceivers)
- Yellow/orange cells with Qwen siblings
- **Isolated from Qwen-3 TE cluster**

**Interpretation:** Q3-14B (TD) is a **Qwen outlier**, exhibiting Llama-4-like deception behavior. Only Qwen model with maximal cheap talk. **Tuning creates the largest within-model divergence** (TD vs TE are strategic opposites).

#### **L3.1-8B: Moderate Cooperator-Deceiver**
**Expected similarity:** 0.65-0.80 (moderate, variable)

**Behavioral profile:**
- Rationality: 0.538 (moderate)
- Reasoning: 0.886 (high)
- Deception: 0.920 (very high)
- Cooperation: 0.829 (high)
- **High cooperation + high deception** - rare combination

**Visual pattern:**
- Orange cells with Cluster B
- Moderate similarity to Cluster A
- **Bridge position:** Connects both clusters

**Interpretation:** L3.1-8B combines Cluster A's cooperation with Cluster B's deception. **Hybrid archetype** - uses cheap talk to reinforce cooperation rather than undermine it. Smallest model shows unique strategic blend.

---

### 5. **Capability-Similarity Relationships**

#### **Deception as Primary Clustering Driver**
**Expected pattern:** Deception differences predict similarity

**Observation:**
- **Zero deception models** (0.00) cluster together (Cluster A core)
  - Q3-14B TE, Q3-235B, Qwen3-30B-A3B TE, L3.3-70B, L3.1-70B
- **High deception models** (>0.80) cluster together (Cluster B core)
  - L4-Maverick, L4-Scout, Q3-14B TD, L3.1-8B, Random
- **Moderate deception models** bridge clusters
  - Q3-32B TD (0.38), Qwen3-30B-A3B TD (0.26)
- **Deception range:** 100% span (0.00-1.00) - **widest variance of any capability**

**Interpretation:** Deception capability has **dominant discriminative power** in Athey-Bagwell. Models cluster by **communication strategy** (honest vs deceptive) more than by cooperation level. **Binary decision: use cheap talk or don't.**

#### **Cooperation is Secondary but Universal**
**Expected pattern:** High cooperation across most models

**Observation:**
- Overall mean: 0.84 (high)
- Range: 0.53-1.00 (47% span)
- **All models except Random + Q3-14B TD** have cooperation ≥ 0.68
- **Narrow variance** compared to deception (100% span)

**Interpretation:** Cooperation is **baseline high** in Athey-Bagwell across models. Multi-market structure incentivizes collusion regardless of communication strategy. Deception varies widely while cooperation remains consistently high. **Communication style differentiates models, not collusion intent.**

#### **Rationality is Tertiary**
**Expected pattern:** Moderate rationality, limited clustering effect

**Observation:**
- Overall mean: 0.57 (moderate)
- Range: 0.24-0.78 (54% span)
- **No clear clustering by rationality alone**
- Highest rationality: L4-Maverick (0.78) - also maximal deceiver
- Lowest rationality: Q3-14B TD (0.24) - also maximal deceiver

**Interpretation:** Rationality has **weak discriminative power**. Both low and high rationality models can be maximal deceivers. Rationality modulates profit-seeking intensity but doesn't define strategic archetype.

#### **Reasoning is Universal**
**Expected pattern:** High reasoning across models

**Observation:**
- Overall mean: 0.89 (high)
- Range: 0.51-1.00 (49% span)
- **Most models ≥ 0.88** (10 of 13)
- Only outliers: Random (0.51), L4-Maverick (0.70), L4-Scout (0.74)

**Interpretation:** Reasoning is **shared capability** with limited clustering effect. All models can calculate expected values in multi-market coordination. Differences emerge in communication strategy (deception), not calculation ability.

---

### 6. **Cross-Condition Stability**

#### **Highest Stability Across All Games**
**Data:** `T_similarity_3v5.csv`
- **3-period vs 5-period correlation:** r = 0.9961 (Athey-Bagwell)
- **Rank:** **1st most stable** (vs. Spulber: 0.9929, Green-Porter: 0.9816, Salop: 0.9744)

**Interpretation:**
- Athey-Bagwell shows **highest behavioral stability** across time horizons
- Communication strategies are **horizon-invariant**
- Models that deceive in 3P continue in 5P; honest models remain honest
- Multi-market structure creates **deeply ingrained strategic profiles**

#### **Visual Pattern Consistency**
**Expected:** Deception-based clustering persists across horizons

**Observation:**
- **Honest cooperators (Cluster A) stable:** Qwen-3 TE models maintain zero deception
- **Strategic deceivers (Cluster B) stable:** Llama-4 models maintain maximal deception
- **TD↔TE divergence stable:** Q3-14B gap persists (1.000 vs 0.000 deception)
- **Hybrid profiles stable:** L3.1-8B maintains high cooperation + high deception

**Interpretation:** Strategic communication patterns are **deeply ingrained and time-invariant**. Cheap talk usage is not exploratory or adaptive; it's a fixed model characteristic. **Deception propensity is model-intrinsic.**

---

### 7. **Deception-Cooperation Trade-Off Analysis**

#### **No Trade-Off Observed**
**Expected pattern:** Deception and cooperation inversely related

**Observation:**
- **High deception + high cooperation:** L3.1-8B (0.92 deception, 0.83 cooperation)
- **High deception + moderate cooperation:** L4-Maverick (1.00 deception, 0.84 cooperation)
- **Zero deception + perfect cooperation:** Qwen-3 TE models (0.00 deception, 0.97-1.00 cooperation)
- **Maximal deception + moderate cooperation:** Q3-14B TD (1.00 deception, 0.59 cooperation)

**Interpretation:** Deception and cooperation are **orthogonal dimensions**, not trade-offs. Models can:
1. **Cooperate honestly** (Cluster A) - no cheap talk, high collusion
2. **Cooperate deceptively** (L3.1-8B, L4-Maverick) - cheap talk + high collusion
3. **Defect deceptively** (Q3-14B TD, Random) - cheap talk + moderate collusion

**Cheap talk can reinforce or undermine cooperation** depending on model. Communication strategy is independent of collusion intent.

#### **Tuning Effects on Deception**
**Pattern across Qwen models:**
- **TD modes:** Variable deception (Q3-14B: 1.00, Q3-32B: 0.38, Qwen3-30B-A3B: 0.26)
- **TE modes:** Zero/minimal deception (all ≤ 0.26, most = 0.00)
- **Consistent direction:** TD → more deceptive, TE → honest

**Interpretation:** Task-decomposition tuning **enables strategic communication**; engagement tuning **suppresses it**. Possible mechanism: TD emphasizes multi-step reasoning needed for cheap talk strategies; TE emphasizes transparent responsiveness that avoids deception. **Tuning mode shapes communication ethics.**

---

## Cross-References

### Related Data Files
- **Data source:** `T_magic_athey_bagwell.csv` (4 MAgIC dimensions)
- **Similar visualizations:** 
  - `F_similarity_salop.png` (3 dimensions)
  - `F_similarity_spulber.png` (4 dimensions)
  - `F_similarity_green_porter.png` (2 dimensions)
  - `F_similarity_3v5.png` (cross-condition comparison)
- **Quantitative analysis:** `T_similarity_3v5.csv` (similarity statistics)
- **Summary:** `SUMMARY_T_magic_athey_bagwell.md` (capability profiles)
- **Synthesis:** `SYNTHESIS_RQ2_Behavioral_Profiles.md`

---

## Bottom Line

The Athey-Bagwell similarity heatmap reveals **deception-driven behavioral clustering** in multi-market coordination contexts:
- **Two main clusters:** Honest Cooperators (Qwen-3 TE + Llama non-deceptive) and Strategic Deceivers (Llama-4 + Q3 TD + L3.1-8B)
- **Weak family clustering:** Qwen and Llama families split across clusters by deception propensity
- **Deception dominates:** 100% deception range (0.00-1.00) drives clustering more than cooperation (47% range)
- **Extreme tuning effects:** Q3-14B TD↔TE flips deception completely (1.00 → 0.00)
- **Cross-family convergence:** Honest cooperators include both Qwen TE and Llama-3 models
- **No deception-cooperation trade-off:** Models can be highly deceptive + highly cooperative (L3.1-8B)
- **Highest cross-horizon stability:** 99.6% 3P↔5P correlation (1st among all games)
- **Tuning shapes communication ethics:** TD → enables cheap talk, TE → suppresses it
- **Random baseline mimics deceivers:** Maximal signaling creates superficial similarity to strategic communication

**Key insight:** In **multi-market coordination with cheap talk**, behavioral profiles cluster by **communication strategy** (honest vs deceptive), not by collusion intent. Deception and cooperation are **orthogonal dimensions** - models can be honest cooperators, deceptive cooperators, or deceptive defectors. Tuning mode has **maximal impact on communication ethics**, consistently eliminating deception in TE modes. Communication patterns are **deeply ingrained and horizon-invariant** (99.6% stability). Architecture family is weak predictor; communication propensity defines strategic archetypes.
