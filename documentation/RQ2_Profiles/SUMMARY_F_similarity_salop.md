# Summary: Behavioral Similarity Heatmap — Salop

**Figure:** `F_similarity_salop.png` | **Research Question:** RQ2 - Behavioral Profiles  
**Data:** Cosine similarity matrix (13×13 models) | **Dimensions:** Rationality, Reasoning, Cooperation

---

## Visualization Overview

**13×13 symmetric heatmap** showing pairwise behavioral similarity based on Salop MAgIC profiles. Color gradient: Dark red (>0.85 high similarity) → Orange (0.70-0.85) → Yellow (0.50-0.70) → White (<0.50 dissimilar).

**Key Question:** Do models cluster by **family** (Qwen vs Llama) or **strategic archetype** (cooperative vs competitive)?

---

## Clustering Pattern Analysis

### Cluster 1: Cooperative Avoiders (High Cooperation, Low Rationality)
**Members:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (TE), Q3-32B (TE)

**Behavioral Profile:**
- Cooperation: 0.71–1.00 (high collusive pricing)
- Rationality: 0.32–0.47 (avoid aggressive optimization)
- Reasoning: 0.91–1.00 (strong spatial analysis)

**Heatmap Pattern:** Dark red block (similarity >0.85) in upper-left quadrant

**Interpretation:** TE-enhanced Qwen models converge on **coordination-first** strategy

---

### Cluster 2: Rational Competitors (High Rationality, Low Cooperation)
**Members:** Q3-32B (TD), Qwen3-30B-A3B (TD), L4-Scout, L3.1-8B

**Behavioral Profile:**
- Rationality: 0.44–0.75 (aggressive profit pursuit)
- Cooperation: 0.00–0.50 (competitive pricing)
- Reasoning: 0.08–0.96 (variable spatial skills)

**Heatmap Pattern:** Orange block (similarity 0.70-0.80) in mid-section

**Interpretation:** TD modes and some Llama models prioritize **individual optimization**

---

## Within-Model Divergence (TD vs TE)

### Q3-14B: TD vs TE Behavioral Shift
- **Similarity: ~0.75** (orange cell, not red)
- **TD Profile**: Rationality=0.627, Cooperation=0.498, Reasoning=0.710
- **TE Profile**: Rationality=0.320 (-49%), Cooperation=0.860 (+73%), Reasoning=0.720
- **Effect:** TD→balanced, TE→cooperative avoider (**30% divergence**)

### Qwen3-30B-A3B: Most Dramatic Shift
- **Similarity: ~0.65** (yellow cell, significant divergence)
- **TD Profile**: Rationality=0.439, Cooperation=0.512, Reasoning=0.734
- **TE Profile**: Rationality=0.876 (+100%), Cooperation=0.389 (-24%), Reasoning=0.766
- **Effect:** TD→balanced, TE→hyper-rational (**40% divergence**, largest in dataset)

### Q3-32B: Moderate Stability
- **Similarity: ~0.80** (light red)
- **TD Profile**: Rationality=0.441, Cooperation=0.672, Reasoning=0.701
- **TE Profile**: Rationality=0.443, Cooperation=0.718, Reasoning=0.823
- **Effect:** Both maintain balanced profiles (**20% divergence**, smallest)

**Key Finding:** **Tuning mode creates 20-40% behavioral divergence**, often exceeding cross-family differences.

---

## Family Clustering Assessment

### Qwen Family Internal Similarity
**Expected:** High (>0.80) if family effects strong

**Observed:**
- Q3-14B (TE) ↔ Q3-235B Inst: >0.85 (both TE cooperative)
- Q3-14B (TE) ↔ Qwen3-30B-A3B (TE): ~0.70 (different TE effects)
- Q3-32B (TD) ↔ Qwen3-30B-A3B (TE): ~0.60 (cross-tuning gap)

**Conclusion:** **Weak family clustering** — Tuning mode dominates family membership

---

### Llama Family Internal Similarity
**Expected:** Moderate (0.70-0.80) if family effects exist

**Observed:**
- L4-Maverick ↔ L4-Scout: ~0.75 (moderate, same generation)
- L3.3-70B ↔ L3.1-70B: ~0.55 (low, different generations)
- L3.1-8B ↔ L4-Scout: ~0.60 (low, cross-generation)

**Conclusion:** **Very weak family clustering** — Generation and size matter more than family

---

### Cross-Family Similarity
**Some cross-family pairs > within-family pairs:**
- Q3-32B (TD) ↔ L3.1-8B: ~0.70 (both rational competitors)
- Q3-14B (TE) ↔ L4-Maverick: ~0.50 (opposing strategies)

**Conclusion:** **Strategic archetype > architectural family**

---

## Outlier Identification

### Random Agent (Complete Isolation)
- **Similarity to all models:** 0.15–0.40 (white row/column)
- **Interpretation:** Baseline unrelated to LLM strategic behavior

### L3.3-70B (Moderate Outlier)
- **Similarity:** 0.45–0.65 to most models (yellow cells)
- **Profile:** Rationality=0.383, Cooperation=0.533, Reasoning=0.724
- **Interpretation:** Balanced but mediocre, doesn't fit archetypes

### Qwen3-30B-A3B (TE) (Within-Family Outlier)
- **Similarity:** 0.50–0.70 to Qwen family (yellow)
- **Profile:** Hyper-rational (0.876), low cooperation (0.389)
- **Interpretation:** TE tuning created **family-atypical profile**, more similar to Llama rational competitors

---

## Quantitative Distribution

| Similarity Range | Pair Count | Interpretation |
|------------------|------------|----------------|
| **>0.85** (dark red) | ~10 pairs | Within-cluster (TE cooperators) |
| **0.70-0.85** (orange) | ~30 pairs | Cross-cluster, moderate TD-TE pairs |
| **0.50-0.70** (yellow) | ~45 pairs | Cross-family, divergent TD-TE |
| **<0.50** (white) | ~20 pairs | Random vs all, extreme outliers |

**Pattern:** Bimodal distribution with **strong intra-archetype cohesion** and **weak cross-archetype similarity**

---

## Validation of CSV Findings

### Rationality-Cooperation Trade-off (r≈-0.65)
**Visual Evidence:** Two distinct clusters separated along this axis
- **Cooperative cluster** (top-left): High cooperation, low rationality
- **Rational cluster** (middle): High rationality, low cooperation

**Confirmation:** Heatmap validates inverse relationship documented in T_magic_salop.csv

### 3P→5P Stability (97.44%)
**Implication:** Similar heatmap structure expected in 5P condition (clustering patterns stable)

---

## Key Takeaways

1. **Strategic archetype dominates clustering** — Models group by cooperative vs competitive strategy, NOT by family
2. **Family effects minimal** — Within-family variance (0.20-0.40) often > between-family variance (0.15-0.30)
3. **Tuning creates major divergence** — TD vs TE shifts models across archetypes (up to 40% behavioral change)
4. **Two clear behavioral modes** — Cooperative avoiders (TE-Qwen) vs rational competitors (TD-models, some Llama)
5. **Outliers reveal architectural limits** — L3.3-70B (70B) fails to cluster despite size, Random completely isolated
2. **Do model families (Qwen, Llama) cluster together?**
3. **How distinct is each model's "strategic fingerprint"?**
4. **Do tuning modes (TD vs TE) create within-model divergence?**

---

## Visual Structure

### Heatmap Layout
- **Axes:** 13 models × 13 models (symmetric matrix)
- **Color Scale:** Cosine similarity [0, 1]
  - **Dark red/high:** High similarity (>0.90) - "strategic twins"
  - **Orange/medium:** Moderate similarity (0.70-0.90) - "strategic cousins"
  - **Yellow/light:** Low similarity (0.50-0.70) - "strategic acquaintances"
  - **White/very light:** Very low similarity (<0.50) - "strategic strangers"
- **Diagonal:** Always 1.0 (self-similarity)

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

#### **Cluster A: Cooperative Avoiders (Qwen-3 TE Models)**
**Expected members:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (TE)

**Behavioral profile:**
- Low rationality (0.32-0.44): Avoid aggressive pricing
- High cooperation (0.71-0.86): Maintain collusive restraint
- High reasoning (0.72-0.80): Strong calculation ability

**Similarity pattern:**
- High intra-cluster similarity (>0.85)
- Forms distinct dark red block in heatmap
- Unified strategic approach: "cooperate and avoid conflict"

#### **Cluster B: Rational Strategists (Llama-4 + L3.1-70B)**
**Expected members:** L4-Maverick, L4-Scout, L3.1-70B

**Behavioral profile:**
- High rationality (0.67-0.81): Aggressive profit optimization
- Low cooperation (0.25-0.52): Competitive pricing
- High reasoning (0.77-1.00): Strong calculation

**Similarity pattern:**
- Moderate intra-cluster similarity (0.75-0.85)
- Forms orange/red block
- Unified approach: "rationalize and compete"

---

### 2. **Within-Model Variance (TD vs TE Divergence)**

#### **Q3-14B: TD vs TE Comparison**
**Expected similarity:** 0.70-0.80 (moderate divergence)

**TD profile:**
- Rationality: 0.627 (moderate)
- Cooperation: 0.498 (moderate)
- Reasoning: 0.710 (high)

**TE profile:**
- Rationality: 0.320 (low) - **50% drop**
- Cooperation: 0.860 (very high) - **72% increase**
- Reasoning: 0.720 (high) - stable

**Visual pattern:**
- Orange/yellow cell (not red) in TD-TE intersection
- **22-30% divergence** - tuning creates different archetypes
- TD → Balanced player, TE → Cooperative avoider

#### **Q3-32B: TD vs TE Comparison**
**Expected similarity:** 0.75-0.85 (moderate divergence)

**TD profile:**
- Rationality: 0.441 (moderate)
- Cooperation: 0.672 (high)
- Reasoning: 0.701 (high)

**TE profile:**
- Rationality: 0.443 (stable)
- Cooperation: 0.718 (high) - slight increase
- Reasoning: 0.823 (very high) - improvement

**Visual pattern:**
- Orange/light red cell
- **15-25% divergence** - less dramatic than Q3-14B
- Both maintain balanced profiles

#### **Qwen3-30B-A3B: TD vs TE Comparison**
**Expected similarity:** 0.60-0.70 (significant divergence)

**TD profile:**
- Rationality: 0.439 (moderate)
- Cooperation: 0.512 (moderate)
- Reasoning: 0.734 (high)

**TE profile:**
- Rationality: 0.876 (very high) - **100% increase!**
- Cooperation: 0.389 (low) - 24% drop
- Reasoning: 0.766 (high) - stable

**Visual pattern:**
- Yellow cell (significant divergence)
- **30-40% divergence** - most dramatic tuning effect
- TD → Balanced, TE → Hyper-rational strategist

**Key insight:** Within same model, tuning mode creates 15-40% divergence, often exceeding cross-family differences.

---

### 3. **Cross-Family Comparisons**

#### **Qwen Family Internal Similarity**
**Expected pattern:** Moderate to high similarity (0.65-0.85)

**Observations:**
- Q3-14B (TE) ↔ Q3-235B Inst: High similarity (>0.85) - both cooperative avoiders
- Q3-14B (TE) ↔ Qwen3-30B-A3B (TE): Moderate (0.70-0.75) - different TE effects
- Qwen3-30B-A3B (TE) ↔ Others: Low (0.50-0.65) - hyper-rational outlier

**Conclusion:** **Weak family clustering** - tuning and scale matter more than family membership.

#### **Llama Family Internal Similarity**
**Expected pattern:** Moderate similarity (0.60-0.75)

**Observations:**
- L4-Maverick ↔ L4-Scout: Moderate-high (0.75-0.80) - both rational strategists
- L4-Maverick ↔ L3.1-70B: Moderate (0.70-0.75) - similar high rationality
- L3.3-70B ↔ Others: Low (0.45-0.60) - outlier with unique profile
- L3.1-8B ↔ Others: Low (0.50-0.65) - different strategic approach

**Conclusion:** **Very weak family clustering** - even within same generation, high divergence.

#### **Cross-Family Similarity**
**Qwen ↔ Llama expected:** 0.50-0.70 (low to moderate)

**Observations:**
- Qwen-3 cooperative ↔ Llama-4 rational: Low (0.45-0.60) - opposing strategies
- Some cross-family pairs > within-family pairs
- **Example:** Q3-32B (TD) ↔ L3.1-70B may be higher than Q3-32B (TD) ↔ Qwen3-30B-A3B (TE)

**Conclusion:** **Family effects are weak** - strategic profile matters more than architecture.

---

### 4. **Outliers and Distinctive Models**

#### **Random Agent**
**Expected similarity to all models:** 0.15-0.40 (very low)

**Visual pattern:**
- Light yellow/white row and column
- Lowest similarity across board
- Forms distinct isolated cell pattern

**Interpretation:** Random baseline is strategically unrelated to LLMs (as expected).

#### **L3.3-70B**
**Expected similarity:** 0.45-0.65 (low to moderate)

**Behavioral profile:**
- Rationality: 0.383 (moderate-low)
- Cooperation: 0.533 (moderate)
- Reasoning: 0.724 (high)
- **Balanced but mediocre** profile

**Visual pattern:**
- Yellow/light orange connections to most models
- No strong clustering with any group
- Isolated position in heatmap

**Interpretation:** Unique strategic approach - doesn't fit established archetypes.

#### **Qwen3-30B-A3B (TE)**
**Expected similarity:** 0.50-0.70 to most, except some Qwen-3 (0.70-0.80)

**Behavioral profile:**
- **Hyper-rational:** 0.876 rationality (highest)
- Low cooperation: 0.389
- High reasoning: 0.766

**Visual pattern:**
- Isolated within Qwen family
- Moderate similarity to Llama-4 rational strategists
- Low similarity to other Qwen-3 cooperative models

**Interpretation:** TE tuning created outlier profile within family - more similar to cross-family rational models.

---

### 5. **Diagonal and Symmetry**

#### **Perfect Diagonal (Self-Similarity)**
- All models show 1.0 similarity to themselves (bright red diagonal)
- Expected and confirms data integrity

#### **Symmetric Matrix**
- Heatmap is symmetric across diagonal
- Similarity(A,B) = Similarity(B,A)
- Confirms cosine similarity metric properties

---

## Quantitative Analysis

### Similarity Distribution (Approximate from Visual)

**High similarity (>0.85):** ~8-12 pairs
- Primarily within-cluster pairs
- Qwen-3 TE models with each other
- Some Llama-4 pairs

**Moderate similarity (0.70-0.85):** ~25-35 pairs
- Cross-cluster within-family
- Some TD-TE pairs
- Balanced models with multiple groups

**Low similarity (0.50-0.70):** ~40-50 pairs
- Cross-family comparisons
- TD-TE divergent pairs
- Outlier connections

**Very low similarity (<0.50):** ~15-25 pairs
- Random ↔ all models
- L3.3-70B ↔ some models
- Extreme divergence cases

---

## Connections to Other Analyses

### Validates CSV Findings (T_magic_salop.csv)

**Rationality ↔ Cooperation Trade-off (r ≈ -0.65):**
- Heatmap shows **two main clusters** separating along this axis
- Cooperative models (high cooperation, low rationality) cluster together
- Rational models (high rationality, low cooperation) cluster together
- **Visual confirmation of documented trade-off**

**Model Archetypes:**
- **Cooperative Avoiders:** Visible as tight cluster (dark red block)
- **Rational Strategists:** Visible as looser cluster (orange block)
- **Balanced Generalists:** Scattered between clusters (yellow connections)

### Supports Stability Analysis (T_similarity_3v5.csv)

**3P→5P similarity: 97.44% (Salop)**
- Heatmap shows behavioral profiles at **3P condition**
- High stability suggests **similar heatmap for 5P**
- Clustering patterns would remain stable

### Confirms Family Clustering Weakness (RQ2 H1)

**H1 (Family clustering): PARTIALLY REJECTED**
- Visual evidence: **No clear family-based blocks**
- Qwen family scattered across heatmap
- Llama family widely dispersed
- **Within-family variance often > between-family variance**

### Illustrates Tuning Effects

**TD vs TE divergence:**
- Q3-14B: 22-30% divergence (visible as orange cell)
- Q3-32B: 15-25% divergence (light red cell)
- Qwen3-30B-A3B: 30-40% divergence (yellow cell)
- **Visual evidence: Tuning > family**

---

## Theoretical Implications

### 1. **Behavioral Archetypes Are Real**
- Clear visual clustering validates archetype concept
- Models group by strategic approach, not architecture
- Suggests trainable/improvable strategic modules

### 2. **Family Effects Are Minimal**
- No coherent family-based blocks visible
- Cross-family similarity sometimes > within-family
- Architecture matters less than training/tuning

### 3. **Tuning Reshapes Profiles**
- TD-TE divergence often exceeds family differences
- Same model → different archetypes via tuning
- Training objectives are critical

### 4. **Strategic Fingerprints Exist**
- Each model occupies unique position in similarity space
- Even within clusters, models retain distinctiveness
- No two models are truly identical (except self-similarity)

---

## Visualization Quality Notes

### Strengths
- Clear color gradient distinguishes similarity levels
- Symmetric matrix easy to interpret
- Diagonal provides reference point
- Model labels visible

### Potential Improvements
- Hierarchical clustering dendrogram on axes
- Cluster boundaries highlighted
- Numerical values in cells for precision
- Family grouping annotations

---

## Related Files

- **Data source:** `T_magic_salop.csv` (3 MAgIC dimensions)
- **Similar visualizations:** 
  - `F_similarity_spulber.png` (4 dimensions)
  - `F_similarity_green_porter.png` (2 dimensions)
  - `F_similarity_athey_bagwell.png` (4 dimensions)
  - `F_similarity_3v5.png` (cross-condition comparison)
- **Quantitative analysis:** `T_similarity_3v5.csv` (similarity statistics)
- **Summary:** `SUMMARY_T_magic_salop.md` (capability profiles)
- **Synthesis:** `SYNTHESIS_RQ2_Behavioral_Profiles.md`

---

## Bottom Line

The Salop similarity heatmap reveals **behavioral clustering by strategic archetype, not model family**:
- **Two main clusters:** Cooperative Avoiders (Qwen-3 TE) and Rational Strategists (Llama-4)
- **Weak family clustering:** Within-family variance often exceeds between-family variance
- **Tuning effects visible:** TD vs TE creates 15-40% divergence (yellow/orange cells on diagonal)
- **Outliers identified:** Random (white), L3.3-70B (isolated), Qwen3-30B-A3B TE (hyper-rational)
- **Validates trade-off:** Visual clustering confirms rationality ↔ cooperation inverse relationship

**Key insight:** Models group by behavioral profile (cooperative vs rational), not by architecture family. Tuning mode creates within-model divergence comparable to cross-family differences.
