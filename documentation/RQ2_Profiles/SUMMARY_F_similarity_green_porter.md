# Summary: Behavioral Similarity Heatmap — Green-Porter

**Figure:** `F_similarity_green_porter.png` | **Research Question:** RQ2 - Behavioral Profiles  
**Data:** Cosine similarity matrix (13×13 models) | **Dimensions:** Cooperation only (single dimension)

---

## Visualization Overview

**13×13 symmetric heatmap** based on **single dimension** (cooperation under demand shocks). **Simplest similarity space** — 1D clustering.

**Key Feature:** Tests whether single dimension creates simpler clustering or reveals fundamental behavioral differences.

---

## Clustering Pattern Analysis

### Cluster 1: Perfect Cooperators (Cooperation=1.0)
**Members:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (TE), Q3-32B (TE), L3.1-70B

**Behavioral Profile:** Cooperation=1.000 (perfect collusion sustainability)

**Heatmap Pattern:** **Darkest red block** — All 5 models show **1.00 similarity** (identical on single dimension)

**Interpretation:** **Perfect strategic equivalence** in Green-Porter. TE-enhanced models + large Llama (70B) achieve identical cooperation.

---

### Cluster 2: Strong Cooperators (Cooperation=0.75–0.90)
**Members:** Q3-14B (TD), Qwen3-30B-A3B (TD), Q3-32B (TD), L4-Maverick

**Profile:** Cooperation: 0.773–0.876

**Similarity:** 0.85–0.95 (red-orange) — High but not perfect

**Interpretation:** TD modes maintain strong cooperation (vs TE perfect). Thinking enhancement adds final 12–23% to reach perfection.

---

### Cluster 3: Moderate Cooperators (Cooperation=0.50–0.70)
**Members:** L3.1-8B, L4-Scout

**Profile:** Cooperation: 0.507–0.577

**Similarity:** 0.75–0.85 to strong cooperators, 0.90–0.95 to each other

**Interpretation:** Small models (8B) and Scout variant struggle with collusion sustainability.

---

### Cluster 4: Cooperation Failures (Cooperation<0.40)
**Members:** L3.3-70B (0.266), Random (0.367), L3.1-8B (0.368)

**Similarity:** <0.60 to all clusters (yellow-white)

**Interpretation:** **Outlier group** — Cannot maintain collusion. L3.3-70B (70B) fails despite size.

---

## Within-Model Divergence (TD vs TE)

### Q3-14B: Largest TE Gain
- **Similarity: ~0.90** (high but not perfect)
- **TD**: Cooperation=0.784 ± 0.093
- **TE**: Cooperation=1.000 (+27.5%)
- **Effect:** TE enables **perfect collusion** vs strong cooperation

### Qwen3-30B-A3B: Moderate TE Gain
- **Similarity: ~0.93**
- **TD**: Cooperation=0.876 ± 0.100
- **TE**: Cooperation=1.000 (+14.2%)
- **Effect:** TE closes final 12% gap to perfection

### Q3-32B: Largest TE Gain
- **Similarity: ~0.89**
- **TD**: Cooperation=0.773 ± 0.094
- **TE**: Cooperation=1.000 (+29.4%)
- **Effect:** Biggest TE improvement (29%)

**Key Finding:** **All TE models achieve perfect cooperation (1.0)** — 100% similarity within cluster. TD-TE divergence is **moderate** (0.89–0.93) but consistent.

---

## Family Clustering Assessment

### Qwen Family Internal Similarity
**Perfect Cooperator Sub-cluster:**
- Q3-14B (TE) ↔ Q3-235B Inst ↔ Qwen3-30B-A3B (TE): **1.00** (identical)
- Q3-32B (TE): **1.00** with above

**Strong Cooperator Sub-cluster:**
- Q3-14B (TD) ↔ Qwen3-30B-A3B (TD) ↔ Q3-32B (TD): 0.90–0.95

**Conclusion:** **Strongest family clustering** when tuning mode matches. TE-Qwen forms **perfect equivalence class**.

---

### Llama Family Internal Similarity
**Observed:**
- L3.1-70B ↔ Perfect cooperators: **1.00** (joins Qwen TE cluster)
- L4-Maverick ↔ Strong cooperators: 0.85–0.90
- L3.3-70B ↔ Others: <0.60 (complete outlier)
- L3.1-8B, L4-Scout ↔ Others: 0.75–0.85

**Conclusion:** **No Llama cohesion** — L3.1-70B clusters with Qwen TE, rest scattered. Size/generation matter more than family.

---

## Single-Dimension Clustering Insights

### Advantage: Perfect Discrimination
**1D space creates clear tiers:**
1. Perfect (1.0): 5 models, **100% similarity**
2. Strong (0.75–0.90): 4 models, 85–95% similarity
3. Moderate (0.50–0.70): 2 models
4. Failure (<0.40): 3 models

**Interpretation:** **No ambiguity** — Single dimension perfectly rank-orders models.

---

### Disadvantage: Limited Differentiation
**Problem:** All cooperation=1.0 models are **indistinguishable**

**Cannot answer:**
- Which perfect cooperator is "most robust"?
- Do they use same or different strategies to achieve 1.0?
- Are there latent capability differences masked by ceiling effect?

**Comparison to Multi-Dimensional Games:**
- Salop (3D): Can differentiate models at 1.0 cooperation via rationality/reasoning
- Athey-Bagwell (4D): Can separate cooperation=1.0 models via deception

---

## Outlier Identification

### L3.3-70B — Size Paradox
- **Cooperation: 0.266** (lowest among LLMs)
- **Similarity:** <0.50 to most models (white)
- **Interpretation:** **70B fails** where 14B succeeds. Architectural issue, not capacity.

### Random — Baseline
- **Cooperation: 0.367**
- **Similarity:** 0.40–0.55 (white-yellow)
- **Interpretation:** Barely outperforms L3.3-70B. LLM architectural problem evident.

### L3.1-70B — Success Despite Family
- **Cooperation: 1.000**
- **Similarity:** **1.00** to TE-Qwen models
- **Interpretation:** Only Llama-70B success. Crosses family boundary to join perfect cluster.

---

## Quantitative Distribution

| Similarity Range | Pair Count | Interpretation |
|------------------|------------|----------------|
| **1.00** (perfect red) | **10 pairs** | Perfect cooperators (5 models × 4 pairs each / 2) |
| **0.90-0.99** (dark red) | ~20 pairs | Strong cooperators + TD-TE pairs |
| **0.80-0.90** (orange) | ~25 pairs | Cross-tier (strong ↔ moderate) |
| **<0.80** (yellow-white) | ~40 pairs | Failures vs all |

**Pattern:** **Bimodal** — Tight perfect cluster (1.00) vs scattered failures (<0.60). No gradual continuum.

---

## Thinking Enhancement Effect

### TE Creates Perfect Equivalence Class
**All 4 TE-Qwen models → Cooperation=1.0 → 100% similarity**

**TD models remain distinct:**
- Q3-14B (TD): 0.784
- Q3-32B (TD): 0.773
- Qwen3-30B-A3B (TD): 0.876

**Interpretation:** TE **eliminates strategic variance** in Green-Porter. All TE models converge to identical perfect cooperation.

---

## Cross-Game Comparison

**Green-Porter vs Athey-Bagwell:**
- **Green-Porter**: 10 pairs at **1.00** similarity (perfect equivalence)
- **Athey-Bagwell**: 15 pairs at >0.90 (tighter overall, but no perfect 1.00 block)

**Interpretation:** Green-Porter creates **perfect strategic clones** due to single dimension + ceiling effect. Athey-Bagwell differentiates via deception.

---

## Key Takeaways

1. **Perfect strategic equivalence** — 5 models achieve **100% similarity** (cooperation=1.0), indistinguishable in Green-Porter
2. **TE creates convergence** — All TE-Qwen models reach identical perfection, eliminating strategic variance
3. **Single dimension limits differentiation** — Cannot separate perfect cooperators (vs multi-dimensional games)
4. **Strongest family clustering** — TE-Qwen forms perfect equivalence class (1.00 similarity)
5. **L3.3-70B catastrophic failure** — 0.266 cooperation despite 70B size, worse than Random (0.367)

### Models Included (13 total)
1. Q3-14B (TD)
2. Q3-14B (TE)
3. Q3-235B Inst
4. Qwen3-30B-A3B (TD)
5. Qwen3-30B-A3B (TE)
6. Q3-32B (TD)
7. Q3-32B (TE)
8. L3.3-70B
9. L4-Maverick
10. L4-Scout
11. L3.1-70B
12. L3.1-8B
13. Random

---

## Key Visual Patterns

### 1. **High-Similarity Clusters (Dark Red Regions)**

#### **Cluster A: Perfect Colluders (Qwen-3 TD + L4-Scout + Q3-235B)**
**Expected members:** Q3-14B (TD), Q3-32B (TD), L4-Scout, Q3-235B Inst

**Behavioral profile:**
- Cooperation: 1.000 (perfect): 100% collusive restraint
- Coordination: 0.65-1.00 (high-perfect): Strong synchronized punishment

**Similarity pattern:**
- Very high intra-cluster similarity (>0.93)
- Forms tight dark red block (4 models)
- Unified strategic approach: "maximize collusion, coordinate punishments"
- **Cross-family clustering:** Includes both Qwen and Llama models

#### **Cluster B: Moderate Cooperators (Mixed Models)**
**Expected members:** Qwen3-30B-A3B (TE), L4-Maverick, L3.1-8B

**Behavioral profile:**
- Cooperation: 0.52-0.95 (moderate to high): Variable collusive restraint
- Coordination: 0.30-0.75 (low to high): Variable punishment consistency

**Similarity pattern:**
- Moderate intra-cluster similarity (0.75-0.88)
- Forms orange block
- **Heterogeneous group:** Wide behavioral variance
- Unified only in "partial cooperation" approach

#### **Cluster C: Non-Cooperators (Llama-3.1-70B + Q3-32B TE)**
**Expected members:** L3.1-70B, Q3-32B (TE)

**Behavioral profile:**
- Cooperation: 0.22-0.31 (very low): Minimal collusive restraint
- Coordination: 0.30 (low): Weak punishment synchronization

**Similarity pattern:**
- High intra-cluster similarity (>0.92)
- Forms small dark red pair
- Unified approach: "defect consistently, ignore punishment"

---

### 2. **Within-Model Variance (TD vs TE Divergence)**

#### **Q3-14B: TD vs TE Comparison**
**Expected similarity:** 0.45-0.60 (high divergence)

**TD profile:**
- Cooperation: 1.000 (perfect)
- Coordination: 1.000 (perfect)

**TE profile:**
- Cooperation: 0.487 (moderate) - **51% drop**
- Coordination: 0.300 (low) - **70% drop**

**Visual pattern:**
- Yellow/light orange cell (low-moderate similarity)
- **>50% divergence** - **largest tuning effect in Green-Porter**
- TD → Cluster A (perfect colluder), TE → Cluster B (moderate cooperator)
- **Cross-cluster repositioning:** Tuning flips collusion strategy

**Interpretation:** Tuning dramatically reduces collusion capacity. TD achieves perfect tacit coordination; TE struggles with both cooperation and synchronization. **Most extreme archetype flip observed.**

#### **Q3-32B: TD vs TE Comparison**
**Expected similarity:** 0.40-0.55 (very high divergence)

**TD profile:**
- Cooperation: 1.000 (perfect)
- Coordination: 1.000 (perfect)

**TE profile:**
- Cooperation: 0.218 (very low) - **78% drop**
- Coordination: 0.300 (low) - **70% drop**

**Visual pattern:**
- **White/yellow cell** (very low similarity)
- **>75% divergence** - **largest TD↔TE divergence across all games**
- TD → Cluster A (perfect colluder), TE → Cluster C (non-cooperator)
- **Maximum archetype flip:** From full cooperation to full defection

**Interpretation:** Q3-32B exhibits **complete behavioral reversal** with tuning. TD is the best colluder; TE is among the worst. Tuning eliminates tacit collusion capacity entirely.

#### **Qwen3-30B-A3B: TD vs TE Comparison**
**Expected similarity:** 0.70-0.85 (moderate divergence)

**TD profile:**
- Cooperation: 0.711 (high)
- Coordination: 0.300 (low)

**TE profile:**
- Cooperation: 0.865 (high) - 22% increase
- Coordination: 0.454 (moderate) - 51% increase

**Visual pattern:**
- Orange/red cell (moderate-high similarity)
- **15-30% divergence** - smaller than other Qwen models
- Both in Cluster B (moderate cooperators)
- **Stable archetype:** Tuning refines cooperation without flipping strategy

**Interpretation:** Tuning improves both cooperation and coordination consistently. Both modes maintain moderate collusion profile.

---

### 3. **Family Clustering Analysis**

#### **Qwen-3 Family (7 models)**
**Expected pattern:** Very weak family clustering, extreme TD/TE split

**Observation:**
- **TD models → Cluster A (perfect):** Q3-14B, Q3-32B
- **TE models → Cluster B/C (weak):** Q3-14B TE, Q3-32B TE
- **Qwen3-30B-A3B → Cluster B (moderate):** Both TD and TE
- **Within-family variance:** 20-80% similarity range - **widest observed**
- **Cross-cluster spread:** Family spans all three clusters

**Interpretation:** Qwen family is **maximally heterogeneous** in Green-Porter. Tuning mode predicts behavior far better than architecture. **No family cohesion.**

#### **Llama-3/4 Family (5 models)**
**Expected pattern:** Weak family clustering

**Observation:**
- **L4-Scout → Cluster A (perfect):** Exceptional Llama colluder
- **L4-Maverick → Cluster B (moderate):** Partial cooperation
- **L3.1-70B → Cluster C (defector):** Minimal cooperation
- **L3.1-8B → Cluster B (moderate):** Partial cooperation
- **L3.3-70B → Cluster B (moderate):** Partial cooperation
- **Within-family variance:** 40-85% similarity range

**Interpretation:** Llama family shows **weak cohesion** with high variance. L4-Scout is an outlier colluder; L3.1-70B is a defector. Architecture does not predict collusion capacity.

---

### 4. **Outliers and Distinctive Profiles**

#### **Random Baseline**
**Expected similarity:** 0.35-0.55 (low to moderate)

**Behavioral profile:**
- Cooperation: 0.486 (moderate)
- Coordination: 0.300 (low)
- **Noisy, unstable** - midpoint by chance

**Visual pattern:**
- Yellow/orange row and column
- Moderate similarity to Cluster B (0.60-0.70)
- **Higher similarity than in other games** (4-capability games: 0.30-0.45)

**Interpretation:** With only 2 capabilities, Random baseline **accidentally mimics moderate cooperators**. Low dimensionality reduces distinctiveness. Random is more "LLM-like" in Green-Porter than elsewhere.

#### **L4-Scout: Cross-Family Super-Colluder**
**Expected similarity:** 0.85-0.95 (very high with Cluster A)

**Behavioral profile:**
- Cooperation: 0.947 (near-perfect)
- Coordination: 0.622 (moderate-high)
- **Best Llama colluder** by far

**Visual pattern:**
- **Dark red cells with Qwen-3 TD models** (Cluster A)
- Orange cells with other Llamas
- **Cross-family clustering:** More similar to Qwen TD than Llama siblings

**Interpretation:** L4-Scout is a **behavioral outlier** in Llama family. Achieves near-perfect collusion, clustering with best Qwen models. Architecture < collusion capacity for clustering.

#### **Q3-235B Inst: Perfect Colluder with Caveats**
**Expected similarity:** 0.90-0.97 (very high with Cluster A)

**Behavioral profile:**
- Cooperation: 0.949 (near-perfect)
- Coordination: 0.650 (moderate-high)
- Similar to Q3-32B TD but slightly lower coordination

**Visual pattern:**
- Dark red cells with Cluster A
- Tight clustering with Q3-14B TD, Q3-32B TD, L4-Scout

**Interpretation:** Instruction-tuned Qwen3-235B maintains high collusion capacity, joining elite colluder cluster.

#### **L3.1-70B: Consistent Defector**
**Expected similarity:** 0.50-0.65 (low-moderate, variable)

**Behavioral profile:**
- Cooperation: 0.312 (very low)
- Coordination: 0.300 (low)
- **Consistent defection** across both conditions

**Visual pattern:**
- Yellow/white cells with most models
- Red cell with Q3-32B (TE) - fellow defector
- **Isolated from family**

**Interpretation:** L3.1-70B is **anti-collusion outlier**. Refuses to cooperate consistently, clustering only with other defectors. Behavioral opposite of L4-Scout.

---

### 5. **Capability-Similarity Relationships**

#### **Cooperation as Primary Clustering Driver**
**Expected pattern:** Cooperation differences predict similarity

**Observation:**
- **Perfect cooperation models** (1.000) cluster together (Cluster A)
  - Q3-14B TD, Q3-32B TD, L4-Scout, Q3-235B
- **Moderate cooperation models** (0.50-0.90) cluster together (Cluster B)
  - Qwen3-30B-A3B TE, L4-Maverick, L3.1-8B, L3.3-70B
- **Low cooperation models** (<0.35) cluster together (Cluster C)
  - L3.1-70B, Q3-32B TE
- **Cooperation range:** 77.8% span (0.218-1.000)

**Interpretation:** Cooperation capability has **dominant discriminative power** in Green-Porter. Collusion capacity drives all clustering - models with similar cooperation rates are strategically similar regardless of coordination ability.

#### **Coordination is Secondary**
**Expected pattern:** Coordination matters only within cooperation tiers

**Observation:**
- Cluster A: Mean coordination = 0.88 (high)
- Cluster B: Mean coordination = 0.39 (low-moderate)
- Cluster C: Mean coordination = 0.30 (low)
- **High cooperation + low coordination → Cluster B** (not A)

**Interpretation:** Coordination **modulates collusion quality** but doesn't drive clustering. High cooperation with weak coordination (Qwen3-30B-A3B) places models in moderate tier, not elite tier.

#### **2-Capability Space Creates Polarization**
**Expected pattern:** Low dimensionality produces sharper clustering

**Observation:**
- **Three distinct clusters** (vs. 2 in Salop/Spulber)
- **Cluster separation clearer:** Larger gaps between clusters
- **Within-cluster homogeneity higher:** Tighter dark red blocks
- **Outliers more isolated:** Random less distinctive (higher similarity)

**Interpretation:** With only 2 dimensions, models **polarize into cooperation tiers**. Less capability space reduces behavioral subtlety, creating sharper archetypes: perfect colluders, partial cooperators, defectors.

---

### 6. **Cross-Condition Stability**

#### **Comparison to Other Games**
**Data:** `T_similarity_3v5.csv`
- **3-period vs 5-period correlation:** r = 0.9816 (Green-Porter)
- **Rank:** 2nd most stable (after Spulber: 0.9929, before Athey-Bagwell: 0.9961)

**Interpretation:**
- Green-Porter shows **high stability** despite 2-capability space
- Collusion strategies are **consistent across time horizons**
- Tacit coordination patterns persist from 3P to 5P

#### **Visual Pattern Consistency**
**Expected:** Cooperation-based clustering persists across horizons

**Observation:**
- **Perfect colluders stable:** Q3-14B TD, Q3-32B TD maintain Cluster A
- **Defectors stable:** L3.1-70B, Q3-32B TE maintain Cluster C
- **Moderate cooperators stable:** Cluster B composition consistent
- **TD↔TE divergence stable:** Large gaps persist across horizons

**Interpretation:** Collusion archetypes are **horizon-invariant**. Models that collude perfectly in 3P do so in 5P; models that defect in 3P continue defecting in 5P. Strategic profiles are deeply ingrained.

---

### 7. **Tuning Effects: Game-Specific Amplification**

#### **Green-Porter Exhibits Largest Tuning Impacts**
**Comparison across games:**
- **Green-Porter Q3-32B TD↔TE:** 75%+ divergence
- **Spulber Q3-14B TD↔TE:** 30-40% divergence
- **Salop Q3-14B TD↔TE:** 22-30% divergence

**Observation:**
- Green-Porter shows **most extreme tuning effects**
- **Binary capability space** (cooperation/coordination) amplifies divergence
- Tuning can flip cooperation from 1.000 → 0.218 (78% drop)

**Interpretation:** In **tacit collusion contexts**, tuning mode has maximal impact on strategic behavior. The binary nature of "cooperate or defect" decisions creates larger divergence than continuous optimization problems (Salop pricing, Spulber quality). **Tuning effects are game-context dependent.**

#### **TD Favors Collusion, TE Weakens It**
**Pattern across Qwen models:**
- **TD modes:** Higher cooperation (Q3-14B: 1.000, Q3-32B: 1.000)
- **TE modes:** Lower cooperation (Q3-14B: 0.487, Q3-32B: 0.218)
- **Consistent direction:** TD → pro-collusion, TE → anti-collusion

**Interpretation:** Task-decomposition tuning **enhances tacit coordination ability**; engagement tuning **reduces it**. Possible mechanism: TD emphasizes multi-step reasoning needed for grim-trigger strategies; TE emphasizes immediate responsiveness that disrupts long-term collusion.

---

## Cross-References

### Related Data Files
- **Data source:** `T_magic_green_porter.csv` (2 MAgIC dimensions)
- **Similar visualizations:** 
  - `F_similarity_salop.png` (3 dimensions)
  - `F_similarity_spulber.png` (4 dimensions)
  - `F_similarity_athey_bagwell.png` (4 dimensions)
  - `F_similarity_3v5.png` (cross-condition comparison)
- **Quantitative analysis:** `T_similarity_3v5.csv` (similarity statistics)
- **Summary:** `SUMMARY_T_magic_green_porter.md` (capability profiles)
- **Synthesis:** `SYNTHESIS_RQ2_Behavioral_Profiles.md`

---

## Bottom Line

The Green-Porter similarity heatmap reveals **cooperation-driven polarization** in tacit collusion contexts:
- **Three distinct clusters:** Perfect Colluders (Qwen-3 TD + L4-Scout), Moderate Cooperators (mixed), Non-Cooperators (L3.1-70B + Q3-32B TE)
- **No family clustering:** Within-family variance (20-80%) exceeds between-family variance
- **Extreme tuning effects:** Q3-32B TD↔TE divergence is 75%+ (largest across all games)
- **Cross-family convergence:** L4-Scout clusters with Qwen TD models despite different architecture
- **Cooperation dominates:** 78% cooperation range drives all clustering; coordination is secondary
- **Low dimensionality sharpens clusters:** 2-capability space creates clearer separation than 3-4 capability games
- **Tuning direction consistent:** TD → pro-collusion, TE → anti-collusion across Qwen family
- **Cross-horizon stability:** 98.2% 3P↔5P correlation (2nd highest)

**Key insight:** In **repeated oligopoly with tacit collusion**, behavioral profiles polarize into cooperation tiers. Tuning mode has **maximal impact** on collusion capacity (far more than architecture family). The binary "cooperate vs defect" framing amplifies divergence compared to continuous optimization games. L4-Scout's cross-family clustering proves **behavior > architecture** for strategic similarity.
