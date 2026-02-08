# Summary: PCA Variance Decomposition Analysis

**Table:** `T6_pca_variance.csv` | **Research Question:** RQ2 - Behavioral Profiles  
**Analysis:** Principal Component Analysis | **Games:** 4 | **Components:** 2-4 per game

---

## Data Overview

PCA decomposes MAgIC behavioral profiles into orthogonal principal components, revealing **intrinsic dimensionality** of strategic behavior. Measures how many independent capability factors explain variance.

**Key Question:** Are MAgIC dimensions redundant, or does each contribute unique information?

---

## Variance Explained by Game

### Salop (3 dimensions: Rationality, Reasoning, Cooperation)

| Component | Variance | Cumulative | Interpretation |
|-----------|----------|------------|----------------|
| **PC1** | **90.65%** | 90.65% | **Dominant** — Single axis explains 91% |
| **PC2** | 6.15% | 96.81% | Minor secondary |
| **PC3** | 3.19% | 100.00% | Negligible |

**Effective Dimensionality:** **~1 dimension**  
**Interpretation:** Salop behavior **nearly one-dimensional**. Likely captures rationality-cooperation trade-off (r≈-0.65 documented). Models vary primarily along competitive ↔ cooperative spectrum.

---

### Spulber (2 dimensions: Rationality, Reasoning)

| Component | Variance | Cumulative | Interpretation |
|-----------|----------|------------|----------------|
| **PC1** | **75.93%** | 75.93% | **Strong** — Primary capability |
| **PC2** | **18.92%** | 94.85% | **Substantial** — Secondary capability |
| **PC3** | 5.10% | 99.95% | Minor |
| **PC4** | 0.05% | 100.00% | Negligible |

**Effective Dimensionality:** **2 dimensions**  
**Interpretation:** **Most complex** behavioral space. PC1 likely reasoning (mean=0.957, universal high), PC2 likely rationality (0.18–0.75 range). Documents **reasoning-rationality decoupling** (understanding ≠ optimization).

---

### Green-Porter (1 dimension: Cooperation only)

| Component | Variance | Cumulative | Interpretation |
|-----------|----------|------------|----------------|
| **PC1** | **91.54%** | 91.54% | **Dominant** — Single dimension |
| **PC2** | 8.46% | 100.00% | Minor |

**Effective Dimensionality:** **~1 dimension**  
**Interpretation:** CSV contains only cooperation → PCA confirms **single meaningful dimension**. PC2 captures noise or within-cooperation nuances (e.g., variance at cooperation≈0.75–0.90 range).

---

### Athey-Bagwell (4 dimensions: Rationality, Reasoning, Deception, Cooperation)

| Component | Variance | Cumulative | Interpretation |
|-----------|----------|------------|----------------|
| **PC1** | **61.77%** | 61.77% | **Moderate** — Primary but not dominant |
| **PC2** | **36.53%** | 98.30% | **Strong** — Critical secondary |
| **PC3** | 1.69% | 99.99% | Negligible |
| **PC4** | 0.01% | 100.00% | Negligible |

**Effective Dimensionality:** **2 dimensions**  
**Interpretation:** **Most balanced** 2D structure (62%-37% split). PC1 likely reasoning+cooperation cluster (both ≈0.9 mean, correlated). PC2 likely deception (binary: 0.0 vs 1.0, orthogonal to others). Captures documented deception-driven clustering.

---

## Cross-Game Dimensionality Summary

| Game | MAgIC Dims | PC1 | PC2 | Effective Dims | Complexity Rank |
|------|------------|-----|-----|----------------|-----------------|
| **Salop** | 3 | 90.65% | 6.15% | **~1** | Simplest |
| **Green-Porter** | 1 | 91.54% | 8.46% | **~1** | Simplest |
| **Athey-Bagwell** | 4 | 61.77% | 36.53% | **2** | Moderate |
| **Spulber** | 2 | 75.93% | 18.92% | **2** | Most Complex |

**Pattern:** Input dimensionality (1–4 MAgIC dimensions) ≠ effective dimensionality (1–2 PCs).

---

## Dimensionality Reduction Efficiency

### Games with High Redundancy (PC1 >90%)
**Salop (90.65%), Green-Porter (91.54%)**
- **Implication:** MAgIC dimensions highly correlated
- **Salop:** 3 dimensions collapse to 1 (rationality-cooperation trade-off)
- **Green-Porter:** Single dimension by design (cooperation only)

---

### Games with Moderate Redundancy (PC1 60–76%)
**Athey-Bagwell (61.77%), Spulber (75.93%)**
- **Implication:** MAgIC dimensions capture distinct strategic facets
- **Athey-Bagwell:** Deception orthogonal to reasoning/cooperation
- **Spulber:** Reasoning-rationality decoupling (understanding vs optimization)

---

## Kaiser Criterion (Eigenvalue >1)

Using rule-of-thumb: **retain components with variance >1/k** (where k = # dimensions)

### Salop (k=3, threshold=33.3%)
- **PC1 (90.65%)**: ✅ Retain
- **PC2 (6.15%)**: ❌ Discard
- **Result:** 1 component

### Spulber (k=2, threshold=50%)
- **PC1 (75.93%)**: ✅ Retain
- **PC2 (18.92%)**: ❌ Discard (but substantive)
- **Result:** 1–2 components (PC2 borderline meaningful)

### Green-Porter (k=1, threshold=100%)
- **PC1 (91.54%)**: ✅ Retain
- **Result:** 1 component

### Athey-Bagwell (k=4, threshold=25%)
- **PC1 (61.77%)**: ✅ Retain
- **PC2 (36.53%)**: ✅ Retain
- **Result:** 2 components

---

## Theoretical Implications

### 1. MAgIC Dimensions Are Correlated
**Evidence:** PC1 explains 62–91% variance (average 79.7%)

**Interpretation:** Behavioral capabilities **not orthogonal**. High reasoning often co-occurs with high cooperation (Athey-Bagwell, Spulber). Rationality trades off with cooperation (Salop).

---

### 2. Effective Behavioral Space Is Low-Dimensional
**Result:** All games reduce to 1–2 effective dimensions

**Implication:** Models vary along **1–2 primary strategic axes**, not independent 4D space. Suggests **common strategic archetypes** rather than unique profiles.

---

### 3. Deception Is Orthogonal Capability
**Athey-Bagwell:** PC2 (36.53%) likely captures deception dimension

**Evidence:** Deception (mean=0.425) uncorrelated with reasoning (0.915) and cooperation (0.875). Binary capability (0.0 vs 1.0) creates second axis.

---

### 4. Game Complexity ≠ Behavioral Complexity
**Spulber (mechanism design):** 2 effective dimensions  
**Salop (spatial competition):** 1 effective dimension

**Paradox:** Hardest game (Salop, lowest win rates) has **simplest** behavioral structure. Suggests difficulty from **optimization challenge**, not strategic complexity.

---

## Validation of Similarity Heatmaps

### Salop: 1D Structure → Linear Clustering
**Prediction:** Models should cluster along single axis (cooperative ↔ competitive)  
**Heatmap Result:** ✅ Confirmed — Two clusters (cooperative avoiders vs rational competitors)

### Athey-Bagwell: 2D Structure → Orthogonal Clusters
**Prediction:** Models should cluster by (reasoning+cooperation) and independently by deception  
**Heatmap Result:** ✅ Confirmed — Deception=0.0 cluster (tight, >0.90) vs deception=1.0 cluster (looser, 0.75–0.85)

---

## Cumulative Variance Thresholds

### 95% Variance Threshold
- **Salop:** 2 components (96.81%)
- **Spulber:** 2 components (94.85%) — just below
- **Green-Porter:** 1 component (91.54%) — below, but single dimension by design
- **Athey-Bagwell:** 2 components (98.30%)

**Standard:** 2 components sufficient for 95% variance in 3/4 games

---

### 99% Variance Threshold
- **Salop:** 3 components (100.00%)
- **Spulber:** 3 components (99.95%)
- **Green-Porter:** 2 components (100.00%)
- **Athey-Bagwell:** 3 components (99.99%)

**Standard:** 3 components capture 99%+ variance in all games

---

## Key Takeaways

1. **Low effective dimensionality** — All games reduce to 1–2 principal components (vs 1–4 input dimensions)
2. **High redundancy in Salop/Green-Porter** — PC1 explains >90%, nearly one-dimensional behavior
3. **Moderate redundancy in Spulber/Athey-Bagwell** — PC1+PC2 explain 95–98%, two-dimensional behavior
4. **Deception is orthogonal** — Athey-Bagwell PC2 (36.53%) captures independent deception capability
5. **Game difficulty ≠ behavioral complexity** — Hardest game (Salop) has simplest structure (1D)

#### **Green-Porter: 2-Dimensional (Simplest)**
- **PC1: 91.54%** (dominant dimension)
- **PC2: 8.46%** (minor secondary)
- **Total components:** 2
- **Cumulative 2-component:** 100.00%

**Interpretation:** Green-Porter is **effectively 1-dimensional** - cooperation and coordination are nearly redundant (r ≈ 0.80 correlation).

#### **Athey-Bagwell: 4-Dimensional (Complex)**
- **PC1: 61.77%** (moderate dominance)
- **PC2: 36.53%** (substantial secondary)
- **PC3: 1.69%** (minor)
- **PC4: 0.01%** (negligible)
- **Total components:** 4
- **Cumulative 2-component:** 98.30%

**Interpretation:** Athey-Bagwell requires **2 strong dimensions** - likely rationality-cooperation (PC1) and deception-reasoning (PC2) reflecting documented trade-offs.

---

### 2. **Variance Explained Thresholds**

#### **90% Variance Threshold (Good Compression)**
- **Salop:** **PC1 alone (90.65%)** ✓
- **Green-Porter:** **PC1 alone (91.54%)** ✓
- **Spulber:** **PC1+PC2 (94.85%)** ✓
- **Athey-Bagwell:** **PC1+PC2 (98.30%)** ✓

**Conclusion:** All games can be represented with **1-2 dimensions** at 90%+ variance.

#### **95% Variance Threshold (High Fidelity)**
- **Salop:** **PC1+PC2 (96.81%)** ✓
- **Green-Porter:** **PC1+PC2 (100%)** ✓
- **Spulber:** **PC1+PC2 (94.85%)** - needs PC3 for 99.95%
- **Athey-Bagwell:** **PC1+PC2 (98.30%)** ✓

**Conclusion:** 3 out of 4 games achieve 95%+ with 2 components; Spulber is most complex.

#### **99% Variance Threshold (Near-Perfect)**
- **Salop:** **PC1+PC2+PC3 (100%)** ✓
- **Green-Porter:** **PC1+PC2 (100%)** ✓
- **Spulber:** **PC1+PC2+PC3 (99.95%)** ✓
- **Athey-Bagwell:** **PC1+PC2+PC3 (99.99%)** ✓

**Conclusion:** All games achieve 99%+ with 2-3 components.

---

### 3. **Game Complexity Ranking**

#### By PC1 Dominance (Lower = More Complex)
1. **Athey-Bagwell:** 61.77% (most multi-dimensional)
2. **Spulber:** 75.93%
3. **Salop:** 90.65%
4. **Green-Porter:** 91.54% (simplest, most 1-dimensional)

**Interpretation:** Games with more capability trade-offs (Athey-Bagwell, Spulber) require more dimensions.

#### By Effective Dimensionality (Components for 95%)
1. **Green-Porter:** 2 components (simplest)
2. **Salop:** 2 components
3. **Athey-Bagwell:** 2 components
4. **Spulber:** 3 components (most complex)

**Interpretation:** Spulber's rationality-judgment-self-awareness-reasoning structure creates highest dimensionality.

#### By PC2 Contribution (Higher = More Secondary Structure)
1. **Athey-Bagwell:** 36.53% (strong second dimension)
2. **Spulber:** 18.92%
3. **Green-Porter:** 8.46%
4. **Salop:** 6.15% (weakest secondary)

**Interpretation:** Athey-Bagwell's deception dimension creates strong second axis; Salop's cooperation-rationality trade-off is nearly linear.

---

### 4. **Cross-Game Consistency**

#### **All Games Show Low Intrinsic Dimensionality**
- **PC1 explains 62-92%** (1 dimension captures majority)
- **PC1+PC2 explain 95-98%** (2 dimensions sufficient for high fidelity)
- **PC3+ explain <5%** (minor refinements)

**Conclusion:** Agent behavioral profiles are **low-dimensional** despite measuring 4+ MAgIC dimensions per game.

#### **Variation in Complexity**
- **Range:** 61.77% (Athey-Bagwell) to 91.54% (Green-Porter) for PC1
- **30 percentage point spread:** Substantial cross-game variation
- **No universal dimensionality:** Each game has unique structure

**Interpretation:** Game-specific capability trade-offs create different dimensional structures.

---

### 5. **Implications for MAgIC Dimensions**

#### **High Redundancy Across Measured Dimensions**
- **Salop measures 3 dimensions** (rationality, reasoning, cooperation) but **PC1 explains 91%** → dimensions highly correlated
- **Spulber measures 4 dimensions** (rationality, judgment, reasoning, self-awareness) but **PC1+PC2 explain 95%** → 2 underlying factors
- **Green-Porter measures 2 dimensions** (cooperation, coordination) and **PC1 explains 92%** → nearly redundant (confirms r ≈ 0.80 correlation)
- **Athey-Bagwell measures 4 dimensions** (rationality, reasoning, deception, cooperation) but **PC1+PC2 explain 98%** → 2 underlying factors

#### **Latent Factor Interpretation**

**PC1 (Primary Factor: 62-92% variance):**
- Likely represents **"Strategic Competence"** - general game-playing ability
- Includes rationality, cooperation, coordination (positively loaded)
- May capture model scale/quality effects

**PC2 (Secondary Factor: 6-37% variance):**
- Likely represents **"Capability Trade-offs"**
  - Salop: Cooperation ↔ Rationality
  - Spulber: Rationality ↔ Judgment/Self-awareness
  - Green-Porter: Minimal (cooperation-coordination synergy)
  - Athey-Bagwell: Deception ↔ Cooperation
- Captures orthogonal strategic choices

**PC3+ (Tertiary Factors: <5% variance):**
- Game-specific nuances
- Measurement noise
- Minor capability interactions

---

### 6. **Statistical Efficiency**

#### **Dimension Reduction Benefits**
From 3-4 measured dimensions → 1-2 principal components:

- **Salop:** 3 → 2 dimensions (33% reduction, 96.81% variance retained)
- **Spulber:** 4 → 2 dimensions (50% reduction, 94.85% variance retained)
- **Green-Porter:** 2 → 1 dimension (50% reduction, 91.54% variance retained)
- **Athey-Bagwell:** 4 → 2 dimensions (50% reduction, 98.30% variance retained)

**Average reduction:** 46% fewer dimensions with 95% average variance retention.

**Benefit:** Simpler models, reduced risk of overfitting, easier interpretation.

---

## Connections to Other Analyses

### Within RQ2 (Behavioral Profiles)

#### **Validates Correlation Findings:**
- **Salop:** 91% PC1 confirms strong rationality-cooperation trade-off (documented in T_magic_salop.csv)
- **Spulber:** 95% PC1+PC2 confirms rationality-judgment inverse + self-awareness axis (T_magic_spulber.csv)
- **Green-Porter:** 92% PC1 confirms cooperation-coordination synergy (r ≈ 0.80 in T_magic_green_porter.csv)
- **Athey-Bagwell:** 62% PC1 + 37% PC2 confirms deception-cooperation trade-off (r ≈ -0.70 in T_magic_athey_bagwell.csv)

#### **Supports Archetype Findings:**
- **Low dimensionality:** Validates distinct model archetypes (e.g., "Honest Colluders" vs "Strategic Deceivers")
- **PC1 dominance:** Confirms primary strategic competence axis separates high/low performers
- **PC2 trade-offs:** Validates secondary capability choices (cooperation vs deception, rationality vs judgment)

### To RQ1 (Performance)
- **T_mlr_features_to_performance.csv:** Shows 56% variance explained by features (improved after multicollinearity fix)
- **T5_magic_to_perf.csv:** Shows 82% variance explained by MAgIC
- **PCA finding:** MAgIC dimensions compress to 1-2 factors → suggests performance prediction can be simplified

**Implication:** Performance regression could use PCA-transformed features for better efficiency.

### To RQ3 (Capability Links)
- **T5_magic_to_perf.csv:** Tests MAgIC→performance links
- **PCA finding:** PC1 (strategic competence) likely strongest performance predictor
- **PC2 (trade-offs) may explain performance heterogeneity** across games

---

## Theoretical Implications

### 1. **Behavioral Profiles Are Simpler Than Measured**
- Measuring 3-4 dimensions per game → only 1-2 underlying factors
- **Measurement redundancy:** 40-95% of variance in single component
- **Implication:** Fewer independent capabilities than MAgIC dimensions suggest

### 2. **Game-Specific Dimensionality**
- **No universal dimensional structure:** 62-92% PC1 range
- **Complex games (Athey-Bagwell, Spulber)** show lower PC1 dominance
- **Simple games (Green-Porter, Salop)** are nearly 1-dimensional

**Conclusion:** Game mechanics determine behavioral complexity.

### 3. **Primary vs Secondary Capabilities**
- **PC1 (Strategic Competence):** Universal, explains 62-92%
- **PC2 (Trade-offs):** Game-specific, explains 6-37%
- **PC3+ (Nuances):** Negligible, <5%

**Implication:** Models differ primarily in general competence, secondarily in strategic choices.

### 4. **Simplification Opportunity**
- **90% variance achievable with 1-2 dimensions**
- **95% variance achievable with 2 dimensions** (3 out of 4 games)
- **Benefits:** Easier visualization, reduced overfitting risk, clearer interpretation

---

## Measurement Details

### PCA Methodology
- **Input:** MAgIC dimension scores (normalized [0,1])
- **Output:** Orthogonal principal components (uncorrelated linear combinations)
- **Variance:** Proportion of total variance explained by each component
- **Cumulative:** Running sum of variance explained

### Component Counts by Game
- **Salop:** 3 components (3 measured dimensions)
- **Spulber:** 4 components (4 measured dimensions)
- **Green-Porter:** 2 components (2 measured dimensions)
- **Athey-Bagwell:** 4 components (4 measured dimensions)

**Rule:** Number of PCs = number of measured dimensions.

### Variance Threshold Guidelines
- **70%:** Rough approximation (acceptable for exploratory analysis)
- **90%:** Good compression (suitable for most analyses)
- **95%:** High fidelity (recommended for publication)
- **99%:** Near-perfect (minimal information loss)

---

## Limitations & Caveats

1. **Linear assumption:** PCA assumes linear relationships; nonlinear trade-offs may not be fully captured
2. **Orthogonality constraint:** Forces uncorrelated components; real capabilities may be obliquely related
3. **Sample size:** Based on 12-13 models per game; larger sample might reveal more structure
4. **Game specificity:** Dimensionality findings don't necessarily generalize to other games
5. **No interpretability guarantee:** PC1/PC2 are mathematical constructs, not necessarily psychologically meaningful

---

## Related Files

- **Data:** `T6_pca_variance.csv`
- **Visualization:** `F_pca_scree.png`
- **Related summaries:**
  - `SUMMARY_T_magic_salop.md` (rationality-cooperation trade-off → explains 91% PC1)
  - `SUMMARY_T_magic_spulber.md` (rationality-judgment inverse → explains PC1+PC2 structure)
  - `SUMMARY_T_magic_green_porter.md` (cooperation-coordination synergy → explains 92% PC1)
  - `SUMMARY_T_magic_athey_bagwell.md` (deception-cooperation trade-off → explains PC2)
  - `SUMMARY_T_similarity_3v5.md` (behavioral stability across conditions)
- **Synthesis:** `SYNTHESIS_RQ2_Behavioral_Profiles.md`

---

## Bottom Line

PCA reveals **agent behavioral profiles are low-dimensional despite multi-dimensional measurement**:
- **1 component captures 62-92%** of variance (primary "strategic competence" factor)
- **2 components capture 95-98%** of variance (adding game-specific trade-offs)
- **Game complexity varies:** Athey-Bagwell (61.77% PC1) most complex, Green-Porter (91.54% PC1) simplest
- **Dimensionality hierarchy:** Green-Porter (2D) < Salop (3D) < Spulber/Athey-Bagwell (4D)
- **Reduction efficiency:** 46% average dimension reduction with 95% variance retention

**Key insight:** Models exhibit 1-2 underlying capability factors, not 4+ independent dimensions. This supports parsimonious behavioral profiling and simplified performance prediction models.
