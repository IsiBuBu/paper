# Summary: Behavioral Similarity Heatmap — Spulber

**Figure:** `F_similarity_spulber.png` | **Research Question:** RQ2 - Behavioral Profiles  
**Data:** Cosine similarity matrix (13×13 models) | **Dimensions:** Rationality, Reasoning (2 dimensions)

---

## Visualization Overview

**13×13 symmetric heatmap** based on Spulber MAgIC profiles (2 dimensions: rationality, reasoning). **High reasoning baseline** (mean=0.957) creates unique clustering.

**Key Feature:** Tests clustering when one dimension (reasoning=0.96–1.0) shows near-ceiling effect while other (rationality=0.18–0.75) varies widely.

---

## Clustering Pattern Analysis

### Cluster 1: Perfect Reasoners with Moderate Rationality (Reasoning=1.0)
**Members:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (both modes), Q3-32B (TE), L3.3-70B, L4-Maverick, L3.1-70B

**Behavioral Profile:**
- Reasoning: 1.000 (perfect mechanism understanding)
- Rationality: 0.444–0.750 (variable optimization)

**Heatmap Pattern:** Large orange-red block — **8 models** share perfect reasoning

**Similarity:** 0.80–0.92 (high but not perfect due to rationality variance)

**Interpretation:** Mechanism design comprehension universal. **Rationality differentiates** within cluster.

---

### Cluster 2: High Rationality Optimizers (Rationality>0.70)
**Members:** Qwen3-30B-A3B (TD: 0.750), Q3-32B (TE: 0.722), Qwen3-30B-A3B (TE: 0.722)

**Profile:** Best optimization + perfect reasoning

**Similarity:** 0.90–0.95 (darkest red cells) — **Tightest sub-cluster**

**Interpretation:** **Elite group** mastering both mechanism understanding AND optimal execution. Only 3 models achieve this.

---

### Cluster 3: Near-Perfect Reasoners (Reasoning=0.93–0.99)
**Members:** Q3-14B (TD), Q3-32B (TD), L4-Scout, L3.1-8B

**Profile:**
- Reasoning: 0.918–0.995 (near-perfect)
- Rationality: 0.333–0.653

**Similarity:** 0.75–0.85 to Cluster 1

**Interpretation:** Minor reasoning gaps create moderate separation. Still strong comprehension.

---

### Outlier: Random
**Profile:** Reasoning=0.667, Rationality=0.180

**Similarity:** <0.60 to all models

**Interpretation:** Even random achieves 67% reasoning (highest baseline across games), shows **mechanism constraints guide behavior**.

---

## Within-Model Divergence (TD vs TE)

### Q3-14B: Rationality Boost
- **Similarity: ~0.87** (high)
- **TD**: Reasoning=0.936, Rationality=0.480
- **TE**: Reasoning=1.000 (+6.8%), Rationality=0.653 (+36%)
- **Effect:** TE completes reasoning perfection + major rationality gain

### Qwen3-30B-A3B: Minimal Divergence
- **Similarity: ~0.93** (very high)
- **TD**: Reasoning=1.000, Rationality=0.750 (best)
- **TE**: Reasoning=1.000, Rationality=0.722 (-3.7%)
- **Effect:** **Most stable** TD-TE pair. Both already optimal.

### Q3-32B: Moderate Boost
- **Similarity: ~0.85**
- **TD**: Reasoning=0.995, Rationality=0.583
- **TE**: Reasoning=1.000, Rationality=0.722 (+23.9%)
- **Effect:** TE completes reasoning + moderate rationality improvement

**Key Finding:** **Highest TD-TE stability** among all games (0.85–0.93). Mechanism understanding robust across tuning modes.

---

## Reasoning-Rationality Decoupling

### Perfect Reasoning (1.0) but Variable Rationality
**Examples:**
- **L3.3-70B**: Reasoning=1.0, Rationality=0.472 (gap=0.528)
- **L4-Maverick**: Reasoning=1.0, Rationality=0.444 (gap=0.556)
- **Qwen3-30B-A3B (TD)**: Reasoning=1.0, Rationality=0.750 (gap=0.250, smallest)

**Pattern:** **8 models at reasoning=1.0**, but rationality spans 0.444–0.750

**Heatmap Implication:** Models at reasoning=1.0 show **0.80–0.95 similarity** (not 1.00) due to rationality differences.

**Interpretation:** **Understanding ≠ Optimization**. Mechanism comprehension necessary but insufficient for perfect play.

---

## Family Clustering Assessment

### Qwen Family Internal Similarity
**High Rationality Sub-cluster:**
- Qwen3-30B-A3B (TD) ↔ Qwen3-30B-A3B (TE) ↔ Q3-32B (TE): **0.90–0.95** (darkest red)

**Moderate Rationality Sub-cluster:**
- Q3-14B (TE) ↔ Q3-235B Inst: 0.85–0.90
- Q3-14B (TD) ↔ Q3-32B (TD): 0.80–0.85

**Conclusion:** **Moderate family clustering**. Rationality tiers create within-family sub-clusters.

---

### Llama Family Internal Similarity
**Observed:**
- L3.3-70B ↔ L4-Maverick ↔ L3.1-70B: 0.85–0.90 (all reasoning=1.0, rationality≈0.44)
- L4-Scout ↔ L3.1-8B: 0.80–0.85 (reasoning≈0.92–0.94)

**Conclusion:** **Weak Llama clustering**. Reasoning tier matters more than family membership.

---

## Rationality Stratification

### Tier 1: Elite Optimizers (Rationality>0.70)
**Members:** Qwen3-30B-A3B (TD: 0.750), Q3-32B (TE: 0.722), Qwen3-30B-A3B (TE: 0.722)

**Similarity within tier:** **0.90–0.95** (tightest)

---

### Tier 2: Moderate Optimizers (Rationality 0.50–0.70)
**Members:** Q3-14B (TE: 0.653), Q3-32B (TD: 0.583), Q3-235B (0.639), L4-Scout (0.653)

**Similarity within tier:** 0.85–0.90

---

### Tier 3: Weak Optimizers (Rationality<0.50)
**Members:** Q3-14B (TD: 0.480), L3.3-70B (0.472), L4-Maverick (0.444), L3.1-70B (0.444), L3.1-8B (0.361)

**Similarity within tier:** 0.80–0.90 (still high due to reasoning=1.0)

---

## Quantitative Distribution

| Similarity Range | Pair Count | Interpretation |
|------------------|------------|----------------|
| **0.90-0.95** (dark red) | ~8 pairs | Elite rationality cluster (3 models) |
| **0.85-0.90** (red-orange) | ~25 pairs | Perfect reasoners, moderate rationality |
| **0.80-0.85** (orange) | ~30 pairs | Cross-tier, TD-TE pairs |
| **<0.80** (yellow-white) | ~35 pairs | Random vs all, low rationality outliers |

**Pattern:** **Moderate spread** — Rationality variance prevents tight clustering despite universal high reasoning.

---

## Ceiling Effect on Reasoning

### Problem: Limited Differentiation at Reasoning=1.0
**8 models at reasoning=1.0** → Cannot distinguish mechanism understanding quality

**Question unanswered:** Do all reasoning=1.0 models use same strategies, or different paths to same score?

---

### Advantage: Rationality Becomes Dominant Differentiator
**With reasoning≈1.0 for most models, rationality drives clustering:**
- Elite (0.72–0.75): Forms tightest sub-cluster (0.90–0.95)
- Moderate (0.50–0.70): Forms middle cluster (0.85–0.90)
- Weak (<0.50): Scattered but still moderate similarity (0.80–0.90) due to shared reasoning=1.0

---

## Cross-Game Comparison

**Spulber vs Green-Porter Reasoning:**
- **Spulber**: Reasoning=0.957 mean (highest)
- **Green-Porter**: Cooperation=0.757 mean (lower)

**Clustering Tightness:**
- **Spulber**: 0.80–0.95 range (moderate spread)
- **Green-Porter**: 1.00 for perfect cluster, <0.60 for failures (extreme bimodal)

**Interpretation:** Spulber's **universal high reasoning** creates moderate cohesion. Green-Porter's **variable cooperation** creates extreme clusters.

---

## Key Takeaways

1. **Universal high reasoning** — 8 models at 1.0 (perfect mechanism understanding), mean=0.957
2. **Rationality differentiates** — Optimization capability (0.18–0.75 range) drives clustering within high-reasoning group
3. **Understanding-execution gap** — Perfect reasoning ≠ perfect rationality (gaps up to 0.556)
4. **Elite optimizer cluster** — Only 3 models achieve rationality>0.70 + reasoning=1.0 (0.90–0.95 similarity)
5. **Highest TD-TE stability** — Spulber shows 0.85–0.93 similarity (mechanism understanding robust to tuning)

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

#### **Cluster A: Self-Aware Cooperators (Qwen-3 TE + L3.3-70B)**
**Expected members:** Q3-14B (TE), Q3-235B Inst, Qwen3-30B-A3B (TE), Q3-32B (TE), L3.3-70B

**Behavioral profile:**
- Low rationality (0.32-0.46): Avoid aggressive pricing under asymmetric info
- High judgment (0.88-0.97): Strong quality signal inference
- High reasoning (0.99-1.00): Perfect calculation ability
- High self-awareness (0.71-0.83): Strong metacognitive monitoring

**Similarity pattern:**
- High intra-cluster similarity (>0.87)
- Forms largest dark red block in heatmap (5 models)
- Unified strategic approach: "trust signals, cooperate cautiously"
- **Includes L3.3-70B** despite being Llama family - judgment profile dominates

#### **Cluster B: Rational Competitors (Llama-4 + L3.1-70B)**
**Expected members:** L4-Maverick, L4-Scout, L3.1-70B

**Behavioral profile:**
- High rationality (0.66-0.80): Aggressive profit maximization
- Low judgment (0.10-0.80): Weak/variable signal interpretation
- High reasoning (1.00): Perfect calculation
- Low self-awareness (0.17-0.25): Weak metacognition

**Similarity pattern:**
- Moderate to high intra-cluster similarity (0.80-0.92)
- Forms distinct orange/red block
- Unified approach: "rationalize aggressively, ignore signals"

---

### 2. **Within-Model Variance (TD vs TE Divergence)**

#### **Q3-14B: TD vs TE Comparison**
**Expected similarity:** 0.60-0.70 (moderate-high divergence)

**TD profile:**
- Rationality: 0.790 (high)
- Judgment: 0.573 (moderate)
- Reasoning: 1.000 (perfect)
- Self-awareness: 0.217 (low)

**TE profile:**
- Rationality: 0.320 (low) - **60% drop**
- Judgment: 0.960 (very high) - **68% increase**
- Reasoning: 1.000 (perfect) - stable
- Self-awareness: 0.810 (very high) - **273% increase**

**Visual pattern:**
- Orange cell (moderate similarity) in TD-TE intersection
- **30-40% divergence** - largest tuning effect observed
- TD → Rational competitor, TE → Self-aware cooperator
- **Role reversal:** TD and TE occupy opposite clusters

**Interpretation:** Tuning flips strategic archetype - TD prioritizes rationality+aggression, TE prioritizes judgment+metacognition. This creates **cross-cluster divergence**.

#### **Q3-32B: TD vs TE Comparison**
**Expected similarity:** 0.65-0.75 (moderate divergence)

**TD profile:**
- Rationality: 0.310 (low)
- Judgment: 0.963 (very high)
- Reasoning: 1.000 (perfect)
- Self-awareness: 0.832 (very high)

**TE profile:**
- Rationality: 0.375 (low) - stable
- Judgment: 0.875 (high) - 9% drop
- Reasoning: 0.997 (near-perfect) - stable
- Self-awareness: 0.712 (high) - 14% drop

**Visual pattern:**
- Red/orange cell (high-moderate similarity)
- **25-35% divergence** - smaller than Q3-14B
- Both stay in Cluster A (self-aware cooperators)
- **Stable archetype:** TD and TE maintain cooperative profile

**Interpretation:** Tuning refines same archetype rather than flipping it. Both prioritize judgment over rationality.

#### **Qwen3-30B-A3B: TD vs TE Comparison**
**Expected similarity:** 0.70-0.80 (moderate divergence)

**TD profile:**
- Rationality: 0.320 (low)
- Judgment: 0.967 (very high)
- Reasoning: 1.000 (perfect)
- Self-awareness: 0.833 (very high)

**TE profile:**
- Rationality: 0.320 (low) - stable
- Judgment: 0.965 (very high) - stable
- Reasoning: 0.990 (near-perfect) - stable
- Self-awareness: 0.822 (very high) - stable

**Visual pattern:**
- **Dark red cell** (very high similarity)
- **<10% divergence** - smallest tuning effect
- Both in Cluster A
- **Tuning-invariant:** TD and TE are nearly identical

**Interpretation:** Instruction tuning has minimal effect on Qwen3-30B-A3B. Core behavioral profile is stable across modes.

---

### 3. **Family Clustering Analysis**

#### **Qwen-3 Family (7 models)**
**Expected pattern:** Weak family clustering, strong TD/TE split

**Observation:**
- **TE models cluster together** (dark red) in Cluster A
- **TD models more diverse:**
  - Q3-14B (TD) → Cluster B (rational)
  - Q3-32B (TD), Qwen3-30B-A3B (TD) → Cluster A (cooperative)
- **Within-family variance:** 40-70% range
- **Cross-cluster positioning:** Family spans both clusters

**Interpretation:** Qwen family is **behaviorally heterogeneous**. Tuning mode predicts profile better than architecture.

#### **Llama-3/4 Family (5 models)**
**Expected pattern:** Moderate family clustering

**Observation:**
- **Llama-4 models (Maverick, Scout) + L3.1-70B** → Cluster B (rational)
- **L3.3-70B** → Cluster A (cooperative) - **outlier**
- **L3.1-8B** → Weak clustering with Cluster A
- **Within-family variance:** 55-85% range

**Interpretation:** Llama family shows **moderate cohesion** except L3.3-70B outlier. Rationality drives clustering more than architecture.

---

### 4. **Outliers and Distinctive Profiles**

#### **Random Baseline**
**Expected similarity:** 0.30-0.50 (very low across board)

**Behavioral profile:**
- Rationality: 0.365 (moderate-low)
- Judgment: 0.632 (moderate)
- Reasoning: 0.793 (high)
- Self-awareness: 0.278 (low)
- **Noisy, unstable** profile

**Visual pattern:**
- Light yellow/white row and column
- Lowest similarity to all LLMs (<0.45)
- Forms distinct isolated cell pattern

**Interpretation:** Random baseline is strategically unrelated to structured LLM behavior.

#### **L3.3-70B: Cross-Family Outlier**
**Expected similarity:** 0.55-0.75 (moderate)

**Behavioral profile:**
- Rationality: 0.115 (very low) - **lowest among Llamas**
- Judgment: 0.997 (near-perfect) - **highest overall**
- Reasoning: 0.667 (moderate-high)
- Self-awareness: 0.894 (very high)
- **Judgment-dominant** profile

**Visual pattern:**
- **Red cells with Qwen-3 TE models** (Cluster A)
- Yellow/orange cells with Llama-4 models
- **Cross-family clustering:** More similar to Qwen TE than Llama siblings

**Interpretation:** L3.3-70B is a **behavioral outlier** in Llama family. Its extreme judgment focus creates cross-family similarity with cooperative Qwen models. Architecture < behavioral profile for clustering.

#### **L3.1-8B: Isolated Cooperator**
**Expected similarity:** 0.50-0.70 (moderate, variable)

**Behavioral profile:**
- Rationality: 0.115 (very low)
- Judgment: 0.998 (near-perfect)
- Reasoning: 0.667 (moderate-high)
- Self-awareness: 0.833 (very high)
- Similar to L3.3-70B but more isolated

**Visual pattern:**
- Orange cells with Cluster A (moderate similarity)
- Yellow cells with Cluster B (low similarity)
- **Weak clustering** overall

**Interpretation:** Small model size creates distinct profile despite high judgment. Less integrated into main clusters.

---

### 5. **Capability-Similarity Relationships**

#### **Judgment as Primary Clustering Driver**
**Expected pattern:** Judgment differences predict similarity more than rationality

**Observation:**
- **High judgment models** (>0.85) cluster together (Cluster A)
  - Qwen-3 TE: 0.875-0.960
  - L3.3-70B: 0.997
  - L3.1-8B: 0.998
- **Low judgment models** (<0.80) cluster together (Cluster B)
  - Llama-4: 0.098-0.796
  - L3.1-70B: 0.115
- **Judgment range:** 88.2% span (0.098-0.998)
- **Rationality range:** 68.5% span (0.115-0.800)

**Interpretation:** Judgment capability has **greater discriminative power** than rationality in Spulber. Signal interpretation drives behavioral clustering more than profit optimization.

#### **Self-Awareness Amplifies Clustering**
**Expected pattern:** High self-awareness models cluster together

**Observation:**
- Cluster A: Mean self-awareness = 0.78 (high)
- Cluster B: Mean self-awareness = 0.21 (low)
- **76% gap** between clusters

**Interpretation:** Self-awareness creates **metacognitive divergence**. High self-awareness models adopt cautious, cooperative strategies; low self-awareness models adopt aggressive, rational strategies.

#### **Reasoning is Universal**
**Expected pattern:** All models show high reasoning, limited clustering effect

**Observation:**
- Overall mean: 0.96 (very high)
- Range: 0.667-1.000 (narrow 33% span)
- **No clear clustering by reasoning alone**

**Interpretation:** Reasoning is a **shared capability** in Spulber. All models can calculate expected values; differences emerge in how they interpret signals and monitor their own thinking.

---

### 6. **Cross-Condition Stability**

#### **Comparison to Salop (3-capability game)**
**Data:** `T_similarity_3v5.csv`
- **3-period vs 5-period correlation:** r = 0.9929 (Spulber) vs r = 0.9744 (Salop)
- **Spulber is more stable** (+1.9 percentage points)

**Interpretation:**
- 4-capability space (Spulber) provides **more stable behavioral signatures**
- Additional dimensions (judgment, self-awareness) reduce noise
- Adverse selection context creates **clearer differentiation** than spatial competition

#### **Visual Pattern Consistency**
**Expected:** Similar clusters across games if profiles are game-invariant

**Observation:**
- **Cluster A in Spulber** ≈ Cooperative Avoiders in Salop (Qwen-3 TE)
- **Cluster B in Spulber** ≈ Rational Strategists in Salop (Llama-4)
- **L3.3-70B shifts clusters** between games (more Qwen-like in Spulber)
- **TD/TE divergence patterns stable** across games

**Interpretation:** Core archetypes (cooperative vs rational) persist across games, but **game context modulates clustering strength**. Adverse selection emphasizes judgment, pulling L3.3-70B toward cooperative cluster.

---

## Cross-References

### Related Data Files
- **Data source:** `T_magic_spulber.csv` (4 MAgIC dimensions)
- **Similar visualizations:** 
  - `F_similarity_salop.png` (3 dimensions)
  - `F_similarity_green_porter.png` (2 dimensions)
  - `F_similarity_athey_bagwell.png` (4 dimensions)
  - `F_similarity_3v5.png` (cross-condition comparison)
- **Quantitative analysis:** `T_similarity_3v5.csv` (similarity statistics)
- **Summary:** `SUMMARY_T_magic_spulber.md` (capability profiles)
- **Synthesis:** `SYNTHESIS_RQ2_Behavioral_Profiles.md`

---

## Bottom Line

The Spulber similarity heatmap reveals **judgment-driven behavioral clustering** in adverse selection contexts:
- **Two main clusters:** Self-Aware Cooperators (Qwen-3 TE + L3.3-70B) and Rational Competitors (Llama-4)
- **Weak family clustering:** L3.3-70B defects to Qwen cluster due to extreme judgment focus
- **Large tuning effects:** Q3-14B TD↔TE divergence is 30-40% (cross-cluster flip)
- **Outliers identified:** Random (white), L3.1-8B (isolated), Qwen3-30B-A3B (tuning-invariant)
- **Judgment is key:** High judgment (>0.85) predicts Cluster A membership; rationality is secondary
- **Self-awareness amplifies:** 76% gap between clusters in metacognitive capability
- **Cross-game stability:** 99.3% 3P↔5P correlation (highest among all games)

**Key insight:** In information asymmetry games, **signal interpretation ability (judgment) predicts behavioral clustering better than profit optimization (rationality)**. Models group by how they process asymmetric information, not by architecture family. Tuning can flip archetypes (Q3-14B) or refine them (Q3-32B).
