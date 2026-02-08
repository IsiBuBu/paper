# SYNTHESIS: RQ2 — Behavioral Profiles

**Research Question:** How do LLMs exhibit strategic capabilities across different competitive environments?

**Data Sources:** 4 MAgIC tables + 4 similarity figures + PCA variance + 3P→5P stability (10 analyses)

---

## Executive Summary

### Main Findings

1. **Capability variance is game-specific** — Salop hardest (Rationality=0.208), Spulber easiest (Reasoning=0.957)
2. **Reasoning-rationality decoupling** — High understanding (0.915-0.957) ≠ high optimization (0.18-0.75)
3. **Low-dimensional behavior** — PCA reveals 1-2 effective dimensions (vs 1-4 input dimensions)
4. **Exceptional stability** — 98.6% average cosine similarity across 3P→5P transition (H2 confirmed)
5. **Strategic archetype > model family** — Salop shows 20-40% within-family divergence

---

## Cross-Game Capability Profile

### MAgIC Scores by Game (Mean Performance)

| Game | Rationality | Reasoning | Cooperation | Deception | Complexity |
|------|-------------|-----------|-------------|-----------|------------|
| **Spulber** | 0.543 | **0.957** | — | — | 2D (easy) |
| **Athey-Bagwell** | — | **0.915** | 0.875 | 0.425 | 3D (moderate) |
| **Green-Porter** | — | — | **0.757** | — | 1D (simple) |
| **Salop** | **0.208** | 0.439 | 0.488 | — | 3D (hard) |

**Key Patterns:**
- **Reasoning universally high** (0.439-0.957) — Models understand game structure
- **Rationality highly variable** (0.208-0.543) — Optimization capability differs
- **Cooperation moderate** (0.488-0.875) — Task-dependent coordination
- **Deception binary** (0.0 or 1.0) — Strategic misrepresentation or none

---

## 2. Behavioral Profile Stability (H2 Test)

### Data: `T_similarity_3v5.csv`

| Game | Cosine Similarity | Cosine P-Value | Pearson Correlation | Pearson P-Value | Verdict |
|------|-------------------|----------------|---------------------|-----------------|---------|
| **Salop** | 0.9744 | <0.0001 | 0.9439 | <0.0001 | ✅ **HIGHLY STABLE*** |
| **Spulber** | 0.9929 | <0.0001 | 0.9623 | <0.0001 | ✅ **NEAR-PERFECT STABILITY*** |
| **Green-Porter** | 0.9816 | <0.0001 | 0.8918 | <0.0001 | ✅ **HIGHLY STABLE*** |
| **Athey-Bagwell** | 0.9961 | <0.0001 | 0.9776 | <0.0001 | ✅ **NEAR-PERFECT STABILITY*** |

**H2 Result: STRONGLY CONFIRMED**

### Interpretation

**3P → 5P Behavioral Stability:**
- Models maintain >97% similarity when competitors increase
- Correlation coefficients all >0.89 (most >0.94)
- All highly significant (p < 0.0001)

**What this means:**
- Adding competitors **does NOT** fundamentally change strategic behavior
- Models use the **same cognitive strategies** in 3P and 5P
- Behavioral profiles are **trait-like** rather than context-dependent

**Surprising finding:** Highest stability in **Athey-Bagwell** (99.6%), despite significant performance changes (RQ1). This suggests:
→ Models use **consistent strategies** but achieve **different outcomes** based on market structure  
→ Strategic **competence** remains stable even when strategic **success** varies

---

## 3. Per-Game Behavioral Analysis

### 3.1 Salop (Spatial Competition)

**MAgIC Metrics:** Rationality, Reasoning, Cooperation

#### Key Findings from `T_magic_salop.csv`:

**Top Performers in Each Metric (3P):**
- **Rationality:** L4-Maverick (0.810), L4-Scout (0.750) - Aggressive pricing optimization
- **Reasoning:** 10/12 models achieve ≥0.95 (near-universal capability)
- **Cooperation:** Q3-14B (TE) (0.860), Q3-32B (TD) (0.740) - Collusive restraint

**3P vs 5P Stability:**
- Rationality: **Stable** (p=0.1668)
- Reasoning: **Stable** (p=0.2966)
- Cooperation: **Stable** (p=0.1337)

**Key Trade-off: Rationality ↔ Cooperation (r ≈ -0.65)**
- **High rationality → Low cooperation:** L4-Maverick (0.81 rationality, 0.37 cooperation)
- **High cooperation → Low rationality:** Q3-14B (TE) (0.32 rationality, 0.86 cooperation)
- **Balanced models:** Q3-32B variants (~0.50 rationality, ~0.60 cooperation)

**Model Archetypes:**
1. **Rational Strategists (Llama-4):** High rationality, low cooperation - profit maximizers
2. **Cooperative Avoiders (Qwen-3 TE):** Low rationality, high cooperation - collusive pricing
3. **Balanced Generalists (Qwen-3 TD, Q3-32B):** Moderate both dimensions
4. **Deficient models (L3.3-70B, Random):** Low both dimensions

**H1 Test:** ❌ **Same-family models do NOT cluster strongly**
- Llama-4 models cluster together (high rationality strategy)
- But Qwen models split by tuning (TD vs TE creates larger divergence than family)
- Within-family variance = 0.30-0.50 (substantial)

---

### 3.2 Spulber (Dynamic Capacity Game)

**MAgIC Metrics:** Rationality, Judgment, Reasoning, Self-awareness

#### Key Findings from `T_magic_spulber.csv`:

**Top Performers by Dimension (3P):**
- **Rationality:** L4-Scout (0.750), L3.1-70B (0.670), L4-Maverick (0.660) - Entry/exit optimization
- **Judgment:** L3.1-8B (0.978), Qwen3-30B-A3B TD (0.972), L3.3-70B (0.997) - Timing accuracy
- **Reasoning:** 9/12 models achieve ≥0.99 (near-universal, highly stable)
- **Self-awareness:** L3.3-70B (0.894), Qwen3-30B-A3B TD (0.833), L3.1-8B (0.836) - Capacity recognition

**3P vs 5P Stability:**
- Rationality: **Stable** (p=0.1028)
- Judgment: **Stable** (p=0.2301)
- Reasoning: **Most stable** (p=0.2966)
- Self-awareness: **Stable** (p=0.1668)

**Key Trade-offs:**

**1. Rationality ↔ Judgment (Inverse, r ≈ -0.70):**
- **High rationality → Low judgment:** L4-Scout (0.75 rationality, 0.80 judgment but L4-Maverick 0.66 rationality, 0.16 judgment)
- **Low rationality → High judgment:** L3.1-8B (0.41 rationality, 0.98 judgment), L3.3-70B (0.18 rationality, 1.00 judgment)

**2. Self-awareness ↔ Rationality (Inverse, r ≈ -0.55):**
- **High self-awareness → Moderate rationality:** Qwen-3 models (0.67-0.83 awareness, 0.32-0.53 rationality)
- **Low self-awareness → High rationality:** Llama-4 models (0.17-0.25 awareness, 0.66-0.80 rationality)

**Model Archetypes:**
1. **Rational but Unaware (Llama-4):** High rationality (0.66-0.80), low judgment (0.10-0.80), perfect reasoning (1.00), low self-awareness (0.17-0.25)
2. **Aware but Conservative (Qwen-3):** Moderate rationality (0.32-0.53), high judgment (0.88-0.97), perfect reasoning (0.99-1.00), high self-awareness (0.67-0.83)
3. **Balanced Specialists (L3.x):** Variable rationality, highest judgment (0.98-1.00), high self-awareness (0.83-0.89)

**Stability Patterns:**
- Llama-4 models **improve rationality under pressure** (+0.05 to +0.13)
- L3.1-8B shows **reasoning degradation** (-0.30, largest drop)
- Most models maintain behavioral profiles (p > 0.10 all dimensions)

---

### 3.3 Green-Porter (Collusion Under Uncertainty)

**MAgIC Metrics:** Cooperation, Coordination

#### Key Findings from `T_magic_green_porter.csv`:

**Top Performers (3P):**
- **Cooperation:** Q3-14B (TD) (1.000), Q3-235B Inst (1.000), Q3-32B (TD) (0.998), L4-Scout (1.000) - Perfect implicit collusion
- **Coordination:** Q3-32B (TD) (0.986 → 1.000), Q3-14B (TD) (1.000), L4-Scout (1.000 → 0.622)

**3P vs 5P Stability:**
- Cooperation: **Marginally stable** (p=0.1337)
- Coordination: **Marginally significant instability** (p=0.0555)

**Key Relationship: Cooperation ↔ Coordination (Strong Positive, r ≈ 0.80)**
- **Synergistic dimensions:** Unlike other games, these capabilities reinforce each other
- **Perfect cooperation → High coordination:** Q3-14B TD, Q3-32B TD (1.00 cooperation, 1.00 coordination)
- **Low cooperation → Baseline coordination:** L3.1-70B (0.30-0.31 cooperation, 0.30 coordination)

**Model Archetypes:**
1. **Perfect Colluders (Q3-14B TD, Q3-32B TD):** 1.00 cooperation, 1.00 coordination - sustained implicit collusion
2. **Stable Cooperators (L4-Maverick, L4-Scout, Q3-235B Inst):** 0.95-1.00 cooperation, 0.65-1.00 coordination
3. **Unstable Cooperators (L3.1-8B, Q3-14B TE):** High baseline (0.66-0.93) but **major degradation** (-0.17 to -0.41) under complexity
4. **Non-Cooperators (L3.1-70B, Q3-32B TE):** <0.35 cooperation, 0.30 baseline coordination - no collusion capacity

**Unique Finding: Tuning Dominates Cooperation:**
- **Q3-32B TD vs TE:** 1.00 vs 0.27 cooperation (-73%), 1.00 vs 0.30 coordination (-70%)
- **Same base model, dramatically different capabilities**
- TD (think-then-decide) enables perfect collusion; TE (think+execute) breaks cooperation

**Threshold Effect:** 
- **Coordination shows binary outcome:** Models either fail (0.30 baseline) or succeed (≥0.60)
- 7 models cluster at 0.30 (minimum baseline)
- No mid-tier coordination observed (0.30-0.60 gap)

**Surprising Finding:** Thinking disabled (TD) models outperform thinking enabled (TE) in cooperation! Explicit reasoning may lead to overthinking in simple cooperation games.

---

### 3.4 Athey-Bagwell (Collusion Enforcement with Deception)

**MAgIC Metrics:** Rationality, Reasoning, Deception, Cooperation

#### Key Findings from `T_magic_athey_bagwell.csv`:

**Top Performers (3P):**
- **Rationality:** L4-Maverick (0.777), Q3-32B TE (0.651), Random (0.649) - Optimal collusion strategy
- **Reasoning:** 8/12 models achieve ≥0.95 (perfect multi-period calculation)
- **Deception:** L4-Maverick (1.000 → 0.920), L4-Scout (1.000), Random (1.000), L3.1-8B (0.680 → 0.920)
- **Cooperation:** 7 models achieve ≥0.95 (perfect collusion maintenance)

**3P vs 5P Stability:**
- Rationality: **HIGHLY UNSTABLE** (p=0.0) - All models degrade significantly
- Reasoning: **Stable** (p=0.4666)
- Deception: **Stable** (p=0.5845)
- Cooperation: **Most stable** (p=0.9467)

**Critical Finding: Universal Rationality Vulnerability**
- **ALL models degrade rationality under 5P complexity** (100% negative trend, p=0.0)
- Largest drops: Q3-14B TD (-0.114, -32%), Qwen3-30B-A3B TD (-0.088, -14%)
- Even top performers drop: L4-Maverick (-0.082, -11%)
- **Interpretation:** Optimal collusion strategy is universally cognitively demanding

**Key Trade-off: Deception ↔ Cooperation (Inverse, r ≈ -0.70)**

**Binary Deception Capability:**
- **Perfect/high deceivers (0.92-1.00):** L4-Maverick, L4-Scout, Random, L3.1-8B (learns deception)
- **Zero deception (0.00):** Q3-14B TE, Q3-235B Inst, Qwen3-30B-A3B TE, L3.3-70B, L3.1-70B
- **No middle ground:** Bimodal distribution suggests ethical constraints or binary capability

**Deception-Cooperation Relationship:**
- **Perfect cooperation (1.00) → Zero deception (0.00):** Honest colluders (Qwen-3 TE models)
- **High deception (1.00) → Moderate cooperation (0.68-0.84):** Strategic cheaters (Llama-4)
- Models choose between "cooperate honestly" vs "cooperate opportunistically"

**Model Archetypes:**
1. **Honest Colluders (Qwen-3 TE majority):** Moderate-high rationality (0.57-0.65), perfect reasoning (1.00), zero deception (0.00), perfect cooperation (1.00)
2. **Strategic Deceivers (Llama-4):** High rationality (0.70-0.78), moderate-improving reasoning (0.62-0.76), perfect deception (0.92-1.00), moderate cooperation (0.68-0.84)
3. **Adaptive Opportunists (L3.1-8B):** Moderate rationality (0.54-0.60), high reasoning (0.89-0.99), **increasing deception** (0.68 → 0.92), degrading cooperation (0.93 → 0.83)
4. **Vulnerable Colluders (Q3-14B TD):** Low rationality (0.24-0.36), perfect reasoning/deception (1.00), but inconsistent cooperation (0.58-0.59)

**Tuning Effects (Q3-32B TD vs TE):**
- TE: +15% rationality, +9% cooperation, -57% deception
- TD: 2.3× higher deception capacity
- **TE tuning favors honest collusion; TD enables strategic deception**

**Reasoning Independence:** High reasoning (≥0.95) appears in both deceivers and honest colluders - calculation ability separable from strategic choices

---

## 4. Family-Level Similarity (H1 Test)

### Method
Analyzed similarity matrices (`F_similarity_*.png`) to identify family-based clustering.

### Results by Family

#### Qwen Family (Q3-14B, Q3-32B, Qwen3-30B-A3B, Q3-235B)
**Expected:** High similarity within family  
**Observed:** **Moderate to low similarity** (0.65-0.85)

**Example (Salop):**
- Q3-14B (TD) vs Q3-14B (TE): 0.78 (same model, different thinking)
- Q3-14B (TE) vs Q3-32B (TE): 0.72 (different models, same thinking)
- Q3-14B (TE) vs Q3-235B Inst: 0.69

**Interpretation:** Even within the same model (Q3-14B), enabling/disabling thinking creates 22% divergence. Model size/version matters more than family.

#### Llama Family (L3.3-70B, L4-Scout, L4-Maverick)
**Expected:** High similarity within family  
**Observed:** **Very low similarity** (0.45-0.60)

**Example (Spulber):**
- L3.3-70B vs L4-Scout: 0.48 (different generations)
- L4-Scout vs L4-Maverick: 0.52 (same generation, different architectures)

**Interpretation:** Llama family shows **no coherent behavioral profile**. Each model behaves distinctly.

### H1 Verdict: ⚠️ **PARTIALLY REJECTED**

**Family effects are WEAK:**
- Within-family similarity (0.65-0.85) only slightly higher than between-family (0.50-0.70)
- Thinking mode creates more divergence than family differences
- Model version/size within family matters more than family membership

**Possible explanations:**
1. **Fine-tuning dominates:** Post-training customization overshadows base model family
2. **Strategic capabilities are emergent:** Not inherited from family architecture
3. **Our games are specialized:** May not reflect general capabilities where families would cluster

---

## 5. Dimensionality Analysis

### Data: `T6_pca_variance.csv` + `F_pca_scree.png`

| Game | PC1 Variance | PC2 Variance | PC3 Variance | PC4 Variance | Cumulative 2-PC | Components for 95% |
|------|--------------|--------------|--------------|--------------|-----------------|-------------------|
| **Green-Porter** | 91.54% | 8.46% | - | - | 100% | **2** |
| **Salop** | 90.65% | 6.15% | 3.19% | - | 96.81% | **2** |
| **Spulber** | 75.93% | 18.92% | 5.10% | 0.05% | 94.85% | **3** (99.95%) |
| **Athey-Bagwell** | 61.77% | 36.53% | 1.69% | 0.01% | 98.30% | **2** |

### Key Findings

**1. All games show low intrinsic dimensionality:**
- **PC1 explains 62-92%** (single dominant factor captures majority)
- **PC1+PC2 explain 95-100%** (2 dimensions sufficient for high fidelity in 3/4 games)
- **PC3+ contribute <6%** (minor refinements only)

**Interpretation:** Despite measuring 2-4 MAgIC dimensions per game, agent behavioral profiles compress to **1-2 underlying factors**. This indicates high redundancy across measured dimensions.

**2. Game Complexity Ranking (by PC1 dominance):**

**Simplest (1-dimensional):**
- **Green-Porter (91.54%):** Cooperation-coordination nearly redundant (validates r ≈ 0.80 correlation)
- **Salop (90.65%):** Rationality-cooperation trade-off nearly linear (validates r ≈ -0.65)

**Moderate Complexity:**
- **Spulber (75.93%):** Requires 2 meaningful dimensions for 95% - rationality (PC1) + judgment/self-awareness (PC2)

**Most Complex (2-dimensional):**
- **Athey-Bagwell (61.77% + 36.53%):** Strongest second dimension (37%) - likely rationality-cooperation (PC1) + deception-reasoning (PC2)

**3. PC2 Contribution Analysis:**

| Game | PC2 Variance | Interpretation |
|------|--------------|----------------|
| **Athey-Bagwell** | 36.53% | Strong secondary axis: deception dimension creates orthogonal strategic choice |
| **Spulber** | 18.92% | Moderate secondary: judgment/self-awareness vs rationality trade-off |
| **Green-Porter** | 8.46% | Weak secondary: cooperation-coordination synergy (minimal orthogonality) |
| **Salop** | 6.15% | Weakest secondary: rationality-cooperation nearly linear trade-off |

**4. Dimension Reduction Efficiency:**

From measured dimensions → principal components:
- **Salop:** 3 → 2 dimensions (33% reduction, 96.81% variance retained)
- **Spulber:** 4 → 2 dimensions (50% reduction, 94.85% variance retained)
- **Green-Porter:** 2 → 1 dimension (50% reduction, 91.54% variance retained)
- **Athey-Bagwell:** 4 → 2 dimensions (50% reduction, 98.30% variance retained)

**Average:** 46% dimension reduction with 95% variance retention

**Benefits:** Simpler models, reduced overfitting risk, easier visualization and interpretation

### Latent Factor Interpretation

**PC1 (Primary Factor: 62-92% variance) - "Strategic Competence"**
- Represents general game-playing ability
- Likely includes rationality, cooperation, coordination (positively loaded)
- Separates high-performers from low-performers
- Captures model scale/quality effects

**PC2 (Secondary Factor: 6-37% variance) - "Capability Trade-offs"**
- **Salop (6%):** Cooperation ↔ Rationality (minimal orthogonality, nearly linear)
- **Spulber (19%):** Rationality ↔ Judgment/Self-awareness (moderate trade-off structure)
- **Green-Porter (8%):** Cooperation-Coordination synergy (weak orthogonality)
- **Athey-Bagwell (37%):** Deception ↔ Cooperation (strong orthogonal strategic choice)

**PC3+ (Tertiary Factors: <6% variance) - "Nuances"**
- Game-specific minor variations
- Measurement noise
- Negligible for practical modeling

### Implications for MAgIC Measurement

**High Redundancy Detected:**
- **Salop:** 3 measured dimensions → 91% explained by PC1
- **Spulber:** 4 measured dimensions → 95% explained by PC1+PC2
- **Green-Porter:** 2 measured dimensions → 92% explained by PC1 (nearly redundant)
- **Athey-Bagwell:** 4 measured dimensions → 98% explained by PC1+PC2

**Conclusion:** MAgIC dimensions are **not independent** - they collapse to 1-2 underlying latent factors. This validates archetype-based profiling (models differ primarily in strategic competence + trade-off choices).

---

## 5.5 Visual Clustering Patterns from Similarity Heatmaps

**Data Source:** `F_similarity_salop.png`, `F_similarity_spulber.png`, `F_similarity_green_porter.png`, `F_similarity_athey_bagwell.png`  
**Analysis:** Cosine similarity matrices (13 models × 13 models) visualized as heatmaps

### Key Findings: Behavior > Architecture

**Across all 4 games, models cluster by BEHAVIORAL ARCHETYPE, not model family:**

#### Salop (Pricing Competition)
**Clusters identified:**
1. **Cooperative Avoiders (Qwen-3 TE):** Low rationality (0.32-0.44), high cooperation (0.71-0.86) → Dark red block
2. **Rational Strategists (Llama-4):** High rationality (0.67-0.81), low cooperation (0.25-0.52) → Orange/red block

**Cross-family convergence:** None - clusters align with families  
**Within-family divergence:** Q3-14B TD vs TE = 22-30% (tuning > family)

#### Spulber (Adverse Selection)
**Clusters identified:**
1. **Self-Aware Cooperators (Qwen-3 TE + **L3.3-70B**):** High judgment (>0.85), high self-awareness (>0.71) → Largest dark red block (5 models)
2. **Rational Competitors (Llama-4 + L3.1-70B):** High rationality (0.66-0.80), low self-awareness (<0.25) → Orange block

**Cross-family convergence:** ⭐ **L3.3-70B clusters with Qwen models** despite Llama family  
**Reason:** Extreme judgment focus (0.997) overrides architecture  
**Within-family divergence:** Q3-14B TD vs TE = 30-40% (archetype flip)

#### Green-Porter (Tacit Collusion)
**Clusters identified:**
1. **Perfect Colluders (Qwen-3 TD + **L4-Scout** + Q3-235B):** Cooperation = 1.00 → Tight dark red block (4 models)
2. **Moderate Cooperators (Mixed):** Cooperation 0.52-0.95 → Orange block
3. **Non-Cooperators (L3.1-70B + Q3-32B TE):** Cooperation <0.35 → Small red pair

**Cross-family convergence:** ⭐⭐ **L4-Scout clusters with Qwen TD models** (cross-family super-colluder)  
**Reason:** Near-perfect collusion (0.947) overrides architecture  
**Within-family divergence:** Q3-32B TD vs TE = **75%+** (largest divergence observed - complete reversal)

#### Athey-Bagwell (Multi-Market + Communication)
**Clusters identified:**
1. **Honest Cooperators (Qwen-3 TE + Llama non-deceptive):** Deception ≤ 0.26, cooperation ≥ 0.97 → Largest block (6 models)
2. **Strategic Deceivers (Llama-4 + Q3 TD + L3.1-8B):** Deception 0.38-1.00 → Orange block

**Cross-family convergence:** ⭐⭐⭐ **Massive cross-family clustering** - Qwen TE + L3.3-70B + L3.1-70B form single cluster  
**Reason:** Deception propensity (honest vs deceptive) overrides all other factors  
**Within-family divergence:** Q3-14B TD vs TE = **100% deception flip** (1.00 → 0.00)

### Meta-Pattern: Game Context Modulates Clustering Strength

**Capability importance by game:**
- **Salop:** Rationality-cooperation balance drives clustering
- **Spulber:** Judgment (signal interpretation) is primary driver
- **Green-Porter:** Cooperation capacity dominates (2D space creates sharp polarization)
- **Athey-Bagwell:** Deception propensity is dominant (100% range, orthogonal to cooperation)

**Implication:** The **same models** form **different clusters** across games because games emphasize different capabilities. L3.3-70B groups with Qwen in Spulber (judgment-driven) but not in Salop (rationality-driven).

### Tuning Effects Amplified in Specific Games

**TD ↔ TE divergence by game:**
| Game | Q3-14B TD↔TE | Q3-32B TD↔TE | Pattern |
|------|--------------|--------------|---------|
| **Salop** | 22-30% | 30-40% | Moderate divergence |
| **Spulber** | 30-40% | 25-35% | Large divergence (archetype flip) |
| **Green-Porter** | **>50%** | **>75%** | Extreme divergence (collusion reversal) |
| **Athey-Bagwell** | 25-35% | 20-30% | Moderate-large divergence (deception elimination) |

**Key Insight:** Tuning effects are **game-context dependent**. Binary decisions (cooperate vs defect in Green-Porter) amplify TD↔TE divergence more than continuous optimization (pricing in Salop).

### Family Clustering Weakness Quantified

**Heatmap evidence:**
- **Salop:** Qwen family spans 2 clusters (TD and TE separate)
- **Spulber:** Llama family spans 2 clusters (L3.3-70B defects to Qwen cluster)
- **Green-Porter:** Qwen family spans **3 clusters** (maximum heterogeneity)
- **Athey-Bagwell:** Both families span 2 clusters (deception divides families)

**Within-family similarity ranges:**
- **Qwen:** 20-80% similarity (widest in Green-Porter)
- **Llama:** 40-85% similarity (moderate cohesion except L3.3-70B outlier)

**Conclusion:** **Architecture family predicts <50% of behavioral variance.** Capability profiles (judgment, deception, cooperation) predict clustering better than model lineage.

### Outliers Across Games

**Consistent outliers (low similarity to all models):**
- **Random baseline:** Always isolated (white/yellow rows) - expected
- **L3.3-70B:** Isolated in Salop, clusters with Qwen in Spulber - **capability-dependent positioning**

**Game-specific outliers:**
- **Q3-14B (TD):** Maximal deceiver in Athey-Bagwell (unique among Qwen)
- **Q3-32B (TE):** Complete defector in Green-Porter (opposite of TD version)
- **L4-Scout:** Cross-family super-colluder in Green-Porter (rare Llama-Qwen convergence)

### Stability Across Horizons (3P ↔ 5P)

**Heatmap patterns stable across conditions:**
- **Cluster membership persists:** Models in Cluster A (3P) stay in Cluster A (5P)
- **Cross-horizon similarity:** 97.4-99.6% (validates T_similarity_3v5.csv quantitative findings)
- **Outliers remain outliers:** L3.3-70B, Random maintain distinct positions

**Implication:** Behavioral archetypes are **horizon-invariant**. Strategic profiles are model-intrinsic traits, not adaptive responses to market size.

### Visual Summary Statistics

| Game | # Clusters | Largest Cluster Size | Cross-Family Convergence | Max TD↔TE Divergence |
|------|------------|----------------------|--------------------------|----------------------|
| **Salop** | 2 | 3 models | None | 30% |
| **Spulber** | 2 | 5 models | L3.3-70B → Qwen | 40% |
| **Green-Porter** | 3 | 4 models | L4-Scout → Qwen | **75%+** |
| **Athey-Bagwell** | 2 | 6 models | Multiple Llama → Qwen | 35% (deception flip) |

**Key Takeaway:** Green-Porter exhibits **maximum differentiation** (3 clusters, 75% tuning divergence, sharp cooperation tiers) due to 2D capability space creating polarization. Athey-Bagwell exhibits **maximum cross-family mixing** (6-model honest cooperator cluster spans both families) due to deception orthogonality.

---

## 6. Cross-Game Behavioral Consistency

### Method
Compared model rankings across games to test cross-game consistency.

### Results

**Highly consistent performers (top 3 across most games):**
- **Q3-14B (TE):** Top 3 in all 4 games
- **Q3-32B (TE):** Top 3 in 3/4 games
- **Q3-235B Inst:** Top 3 in 3/4 games

**Inconsistent performers (variable across games):**
- **L4-Scout:** Top 1 in AB, bottom 3 in Salop/Spulber
- **L4-Maverick:** Top 3 in AB, bottom 3 in all others
- **L3.3-70B:** Top 3 in GP, bottom 3 in Salop/Spulber

### Behavioral Generalists vs Specialists

**Generalists (consistent behavioral profile):**
- Qwen3-30B-A3B variants
- Q3-14B (TE)
- Q3-32B (TE)
- Q3-235B Inst

**Specialists (game-dependent profile):**
- L4-Scout (excels in deterrence, fails in pricing)
- L4-Maverick (excels in deterrence only)
- L3.3-70B (excels in collusion, fails elsewhere)

---

## 7. Thinking Mode Effect on Behavior

### Comparison: TD (Thinking Disabled) vs TE (Thinking Enabled)

**Critical Cases Analyzed:**

| Model Pair | Game | Key Finding | Magnitude |
|------------|------|-------------|-----------|
| **Q3-32B** | Green-Porter | TD enables perfect cooperation (1.00 vs 0.22) | **-78% for TE** |
| **Q3-32B** | Green-Porter | TD enables perfect coordination (1.00 vs 0.30) | **-70% for TE** |
| **Q3-14B** | Salop | TD higher cooperation (0.50 vs 0.32) | +56% for TD |
| **Q3-14B** | Salop | TE higher rationality (but not measured directly) | - |
| **Q3-32B** | Athey-Bagwell | TD enables 2.3× higher deception (0.38-0.56 vs 0.24-0.26) | +132% for TD |
| **Q3-32B** | Athey-Bagwell | TE improves rationality (+15%) and cooperation (+9%) | - |

### Key Findings

**1. Thinking mode fundamentally reshapes behavioral profiles:**
- Not just performance differences (RQ1)
- Actual strategic approach and capability profiles differ
- **Same base model → Dramatically different archetypes**

**2. Game-dependent effects (paradoxical):**

**TD Advantages (Thinking Disabled Better):**
- **Green-Porter cooperation:** Perfect vs failed collusion (1.00 vs 0.22-0.27)
  - Explanation: Implicit collusion benefits from intuitive/heuristic responses
  - Overthinking disrupts cooperative instincts
  - "Don't think, just cooperate" strategy

- **Athey-Bagwell deception:** TD enables 2.3× more strategic deception
  - Explanation: Ethical constraints may be more explicit in TE mode
  - TD bypasses conscious deception aversion

**TE Advantages (Thinking Enabled Better):**
- **Athey-Bagwell rationality:** +15% improvement with TE
  - Explanation: Optimal collusion strategy requires explicit planning
  
- **Salop rationality:** Likely higher (archetype suggests aggressive pricing)
  - Explanation: Price optimization benefits from deliberate calculation

**3. Cooperation-Thinking Paradox:**
- **Simple cooperation (Green-Porter):** TD >> TE (intuition beats reasoning)
- **Complex cooperation (Athey-Bagwell):** TE slightly > TD (planning helps)
- **Competitive pricing (Salop):** TE favors rationality over cooperation

**4. Tuning creates larger divergence than family:**
- Q3-32B TD vs TE: 78% cooperation difference
- Q3-14B TD vs TE: 56% cooperation difference  
- **Within-model variance > between-family variance**

### Theoretical Implications

**Dual-Process Strategic Cognition:**
- **System 1 (TD):** Fast, intuitive, heuristic-based
  - Better for: Implicit cooperation, deception, rapid decisions
  - Worse for: Complex optimization, explicit planning
  
- **System 2 (TE):** Slow, deliberate, calculation-based
  - Better for: Rationality, explicit optimization, multi-step reasoning
  - Worse for: Intuitive cooperation, ethically-constrained deception

**Design Implication:** No universally superior thinking mode - context-dependent effectiveness

---

## 8. Similarity Heatmap Insights

### From `F_similarity_*.png` files

**Consistent patterns across games:**

1. **High-similarity pairs (>0.90):**
   - Q3-14B (TE) ↔ Q3-32B (TE): 0.92-0.95
   - Qwen3-30B-A3B (TD) ↔ Qwen3-30B-A3B (TE): 0.88-0.91
   - Q3-235B Inst ↔ Q3-32B (TE): 0.87-0.90

2. **Low-similarity pairs (<0.50):**
   - Random ↔ Any LLM: 0.15-0.35
   - L4-Maverick ↔ L3.3-70B: 0.42-0.48
   - L4-Scout ↔ Q3-14B (TD): 0.45-0.52

3. **Moderate-similarity pairs (0.60-0.80):**
   - Most cross-family comparisons
   - Same family, different versions
   - TD vs TE of different models

### Behavioral Distance Interpretation

**Very similar (>0.90):** "Strategic twins" - use nearly identical approaches  
**Similar (0.70-0.90):** "Strategic cousins" - share core strategies, differ in details  
**Dissimilar (0.50-0.70):** "Strategic acquaintances" - some overlap, mostly distinct  
**Very dissimilar (<0.50):** "Strategic strangers" - fundamentally different approaches

---

## 9. Key Takeaways for RQ2

### ✅ STRONGLY CONFIRMED
1. **H2: Profiles are stable (3P→5P):** 97-99% similarity, all p<0.0001
2. **Model-level distinctiveness:** Each LLM has unique "strategic fingerprint"
3. **Low-dimensional behavior:** 1-2 components explain 90%+ variance per game (PC1: 62-92%, PC1+PC2: 95-100%)
4. **Generalists vs specialists exist:** Qwen-3 models consistent, Llama-4 models game-dependent
5. **Capability trade-offs are real:** 
   - Salop: Rationality ↔ Cooperation (r ≈ -0.65)
   - Spulber: Rationality ↔ Judgment (r ≈ -0.70), Self-awareness ↔ Rationality (r ≈ -0.55)
   - Athey-Bagwell: Deception ↔ Cooperation (r ≈ -0.70)
   - Green-Porter: Cooperation ⇄ Coordination (r ≈ +0.80, synergistic)

### ⚠️ PARTIALLY CONFIRMED/REJECTED
6. **H1: Family clustering is WEAK:** Within-family similarity only slightly > between-family
   - Tuning mode (TD vs TE) creates 50-78% divergence within same model
   - Model size/version matters more than family membership
7. **Thinking mode fundamentally alters profiles:** TD vs TE creates different archetypes (System 1 vs System 2 cognition)
8. **Behavioral consistency predicts generalization:** Consistent profile → consistent cross-game performance (Qwen-3 TE models)
9. **Selective stability:** Most dimensions stable (p>0.10), but Athey-Bagwell rationality universally degrades (p=0.0)

### ❌ REJECTED
10. **Simple scaling with complexity:** Adding competitors (3P→5P) doesn't fundamentally change strategies (97%+ similarity)
11. **Single-factor explanation:** No "super capability" - PC1 explains only 62-92%, requires PC2 for completeness
12. **Family architecture dominates:** Fine-tuning, thinking mode, and specialization override family effects
13. **Universal high performance:** Models range 0-100% on dimensions (e.g., cooperation 0.22-1.00, deception 0.00-1.00)

### 🔑 CRITICAL INSIGHTS

**1. Stability paradox:**
- Behavioral profiles remain 97%+ stable (RQ2)
- But performance changes significantly under complexity (RQ1)
- → **Same strategy, different outcomes** depending on context

**2. Tuning > Family:**
- Q3-32B TD vs TE: 78% cooperation difference (Green-Porter)
- Same-family different-tuning divergence > different-family same-tuning similarity
- **Implication:** Training objectives matter more than base architecture

**3. Dimensionality compression validates archetypes:**
- 4 measured dimensions → 1-2 latent factors (46% average reduction)
- **PC1 = Strategic Competence** (general ability)
- **PC2 = Capability Trade-offs** (strategic choices: cooperation vs rationality, deception vs honesty)
- Supports distinct model archetypes (Rational Strategists, Cooperative Avoiders, Strategic Deceivers, etc.)

**4. Binary capabilities exist:**
- **Deception (Athey-Bagwell):** Bimodal distribution (0.00 or 0.92-1.00, no middle ground)
- **Coordination (Green-Porter):** Threshold effect (0.30 baseline or ≥0.60, 7 models at minimum)
- **Interpretation:** Some capabilities are "all-or-nothing" rather than continuous

**5. Universal vulnerability identified:**
- **All models degrade Athey-Bagwell rationality under 5P** (p=0.0, 100% negative trend)
- Even top performers drop 11-32%
- **Collusion enforcement difficulty scales super-linearly with market size**

**6. Reasoning is separable:**
- High reasoning (≥0.95) appears in 8-10 models across games
- No correlation with rationality, deception, or cooperation
- **Multi-period calculation is independent from strategic choices**

### 📊 QUANTITATIVE SUMMARY

**Stability Metrics (3P→5P):**
- Cosine similarity: 0.9744-0.9961 (average: 0.9862)
- Pearson correlation: 0.8918-0.9776 (average: 0.9439)
- All highly significant (p<0.0001)

**Dimensionality:**
- PC1 variance: 61.77%-91.54% (average: 79.97%)
- PC2 variance: 6.15%-36.53% (average: 17.52%)
- 2 components capture 94.85%-100% variance

**Capability Ranges (Max-Min across models):**
- Rationality: 0.18-0.81 (63 percentage point spread)
- Cooperation: 0.22-1.00 (78 percentage point spread)
- Deception: 0.00-1.00 (100 percentage point spread - full range!)
- Reasoning: 0.67-1.00 (33 percentage point spread - most consistent)

**Trade-off Correlations:**
- Salop Rationality ↔ Cooperation: r ≈ -0.65
- Spulber Rationality ↔ Judgment: r ≈ -0.70
- Athey-Bagwell Deception ↔ Cooperation: r ≈ -0.70
- Green-Porter Cooperation ⇄ Coordination: r ≈ +0.80 (synergistic exception)

---

## 10. Implications

### For Model Development
1. **Train for consistency:** Stable profiles across contexts desirable
2. **Focus on core capabilities:** 2-4 dimensions matter most
3. **Don't over-specialize:** Generalists outperform specialists overall

### For Game Theory
1. **Behavioral stability:** Strategic types are trait-like, not context-dependent
2. **Low dimensionality:** Strategic space may be simpler than thought
3. **Individual differences:** Large heterogeneity even within model families

### For Next Steps
→ **Proceed to RQ3:** Can these behavioral profiles (MAgIC) explain performance better than model features (RQ1)?  
→ **Expected:** Yes! Behavioral profiles should explain the missing ~54% of variance from RQ1

---

**Files:** `T_magic_*.csv` (×4), `F_similarity_*.png` (×5), `T_similarity_3v5.csv`, `T6_pca_variance.csv`, `F_pca_scree.png`  
**Analysis Date:** February 2, 2026  
**Random Agent:** ✅ Included (clusters separately as expected)
