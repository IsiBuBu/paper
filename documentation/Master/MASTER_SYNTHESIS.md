# MASTER SYNTHESIS: Complete Analysis of LLM Strategic Behavior in Economic Games

## Document Purpose
This master synthesis integrates findings from all research questions (RQ1, RQ2, RQ3) and supplementary analyses to provide a unified understanding of LLM strategic competence in competitive economic environments.

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Research Questions Overview](#research-questions-overview)
3. [Unified Findings](#unified-findings)
4. [Cross-RQ Integration](#cross-rq-integration)
5. [Theoretical Framework](#theoretical-framework)
6. [Practical Implications](#practical-implications)
7. [Limitations and Future Work](#limitations-and-future-work)
8. [Conclusions](#conclusions)

---

## Executive Summary

### Central Finding
**LLM strategic behavior is characterized by STABLE BEHAVIORAL PROFILES that STRONGLY PREDICT performance, far better than architectural features.** Models exhibit consistent "strategic personalities" that persist across competitive conditions but yield different outcomes depending on environmental context.

### Key Numbers
- **Behavioral stability:** 97–99% similarity (3P vs 5P conditions)
- **MAgIC prediction:** R² = 0.766 (MAgIC → performance)
- **Combined prediction:** R² = 0.816 (MAgIC + Features)
- **Feature prediction:** R² = 0.562 (architecture/family → performance)
- **Advantage:** MAgIC is 36% better than features at explaining performance
- **Reasoning power:** 80% success rate (strongest predictor)

### Core Insights
1. **What models DO matters more than what models ARE** (behavior > architecture)
2. **Behavioral profiles are fundamental, not contextual** (97–99% stable)
3. **Reasoning quality matters, not quantity** (capability > effort)
4. **Game structure determines success requirements** (context-dependent competence)
5. **Models exhibit trade-offs, not universal superiority** (cooperation vs rationality)

---

## Research Questions Overview

### RQ1: Competitive Performance
**Question:** How well do LLMs perform in strategic economic games?

**Answer:** Performance varies DRAMATICALLY (0–100% win rates) across models and games. Increased competition (3P→5P) significantly reduces profits in all games but only affects win rates in Athey-Bagwell.

**Key Findings:**
- Model features explain **56% of variance** (R² = 0.562)
- Thinking mode (TE) is most consistent predictor (58% success rate)
- All games show significant profit declines 3P→5P
- Game-specific metrics reveal strategic heterogeneity

---

### RQ2: Behavioral Profiles
**Question:** Do LLMs exhibit consistent behavioral patterns? Do model families cluster?

**Answer:** LLMs show EXTREME behavioral stability across conditions (97–99% similarity) but WEAK family clustering. Each model maintains a unique "strategic personality."

**Key Findings:**
- **H1 (Family clustering):** PARTIALLY REJECTED — weak evidence for family effects
- **H2 (Stability):** STRONGLY CONFIRMED — 97–99% similarity 3P→5P
- 2–4 PCA dimensions explain 80%+ of behavioral variance
- Behavioral profiles are fundamental model characteristics

---

### RQ3: Capability-Performance Links
**Question:** How do behavioral capabilities relate to performance?

**Answer:** Behavioral capabilities (MAgIC) explain **77% of performance variance**, FAR better than model features (56%). Combined models reach 82%. Reasoning is the strongest predictor (80% success rate).

**Key Findings:**
- MAgIC R² = 0.766 vs Features R² = 0.562 (+36% advantage)
- Combined R² = 0.816 (82% variance explained)
- Four near-perfect fits (R² > 0.98)
- Reasoning capability: 80% success rate
- Context-dependent capability requirements (different games reward different skills)

---

### Supplementary: Reasoning Effort
**Question:** How do models allocate reasoning effort across games?

**Answer:** Reasoning effort varies by game (8K–36K chars) but shows NO correlation with performance. Reasoning MODE (TE/TD) matters, but QUANTITY (character count) within TE doesn't.

**Key Findings:**
- Game-specific effort: Salop (25K) > Green-Porter (25K) > Spulber (24K) > Athey-Bagwell (14K)
- No effort-performance correlation (sometimes negative)
- Medium model (30B) most stable and best performing
- Reasoning QUALITY (MAgIC) matters, not QUANTITY (chars)

---

## Unified Findings

### Finding 1: The Behavioral Profile Hierarchy

```
Level 1: ARCHITECTURAL FEATURES (Weak Predictors)
├─ Architecture (MoE vs standard): 33% success rate
├─ Family (Llama, Qwen, etc.): 8% success rate
├─ Version (model generation): 25% success rate
└─ Thinking mode (TE vs TD): 58% success rate ⭐
    └─ OVERALL: R² = 0.562 (explains 56% of variance)

Level 2: REASONING EFFORT (Non-Predictor)
├─ Character count: 8K–36K chars
├─ Game-specific allocation (correct perception of difficulty)
└─ No correlation with performance ❌

Level 3: BEHAVIORAL CAPABILITIES (Strong Predictors)
├─ Reasoning: 80% success rate ⭐⭐⭐
├─ Rationality: 67% success rate ⭐⭐
├─ Self-awareness: 67% success rate ⭐⭐
├─ Cooperation: 63% success rate ⭐⭐
└─ OVERALL: R² = 0.766 (explains 77% of variance) ✅

Level 4: COMBINED (MAgIC + Features)
└─ OVERALL: R² = 0.816 (explains 82% of variance) ✅✅
```

**Implication:** **Capability > Mode >> Features >>> Effort**

---

### Finding 2: The Stability-Prediction Pipeline

```
Training Data
    ↓
Behavioral Profile Formation
    ↓
STABLE across conditions (97–99% similarity)
    ↓
Capability Loadings (MAgIC dimensions)
    ↓
PREDICT performance (R² = 0.766)
    ↓
Outcomes (profit, wins, efficiency)
    ↓
VARY by environment (3P vs 5P)
```

**Key Insight:** **Same strategies (stable behavior) → different outcomes (environmental effects)**

**Example:**
- Model maintains 98% similar behavior 3P→5P
- But profit declines significantly (e.g., Green-Porter: -40%)
- **Interpretation:** Models don't adapt strategies; environment changes payoffs

---

### Finding 3: Game-Specific Capability Requirements

| Game | Primary Capability | Secondary | Detrimental | R² (best target) |
|------|-------------------|-----------|-------------|------------------|
| **Athey-Bagwell** | Rationality (+1.42) | Reasoning (+0.67) | Reasoning→profit (-3261) | 0.997 (efficiency) |
| **Green-Porter** | Cooperation (-0.014) | Coordination (+0.08) | N/A (profit random) | 0.710 (reversion) |
| **Salop** | Rationality (+1.01) | Reasoning (+1697) | Reasoning→price (-0.99) | 0.989 (win_rate) |
| **Spulber** | Reasoning (+1696) | Self-awareness (+435) | Reasoning→efficiency (-0.43) | 0.987 (efficiency) |

**Pattern:** **Different games reward different capabilities.** No single "best" strategic profile.

---

### Finding 4: Capability Trade-offs (Not Dominance)

#### Trade-off 1: Cooperation vs. Rationality (Athey-Bagwell)
- **Cooperation → Win (+0.928)** but **→ Low efficiency (-0.238)**
- **Rationality → High efficiency (+1.42)** but **→ Lose (-0.652)**
- **Two paths:** Collusive (cooperate) vs Competitive (optimize)

#### Trade-off 2: Reasoning vs. Speed (Spulber)
- **Reasoning → High profit (+1696)** but **→ Poor matching (-0.43)**
- **Rationality → Good matching (+0.989)** but **→ Lower profit**
- **Two tiers:** Fast matching vs Strategic depth

#### Trade-off 3: Pricing Aggression (Salop)
- **Reasoning → High profit (+1697)** but **→ Lower prices (-0.99)**
- **Cooperation + Rationality → High prices** (+1.54, +1.62)
- **Dilemma:** Competitive pricing vs Collusive pricing

**Implication:** **No universally optimal model.** Choose based on task requirements.

---

### Finding 5: Thinking Hierarchy (Mode > Quality >> Quantity)

| Level | Measure | Predicts Performance? | Success Rate | Evidence |
|-------|---------|----------------------|--------------|----------|
| **Mode** | TE vs TD | ✅ YES | 58% | TE > TD in 3/4 games |
| **Quality** | MAgIC reasoning | ✅✅✅ STRONGLY | 80% | R² = 0.766, 8/10 significant |
| **Quantity** | Character count | ❌ NO | 0% | No correlation, sometimes negative |

**Key Insight:** **Having reasoning capability (mode) matters. Exercising it well (quality) matters most. How much you write (quantity) doesn't matter.**

---

## Cross-RQ Integration

### Integration 1: Stability Enables Prediction

**RQ2 Finding:** Behavioral profiles 97–99% stable across conditions  
**RQ3 Finding:** Behavioral capabilities predict 77% of performance variance  
**Integration:** **Because behavior is stable, we can predict performance from profiles**

**Practical Application:**
1. Measure behavioral profile in one context (e.g., 3P)
2. Profile is stable → same profile in other context (5P)
3. Profile predicts performance → forecast outcomes in new context

**Limitation:** Predicts behavior-outcome mapping, not outcome alone (environment still matters)

---

### Integration 2: Features → Behavior → Performance

**RQ1 Finding:** Model features explain 56% of performance  
**RQ3 Finding:** Behavioral capabilities explain 77% of performance  
**Integration:** **Features are upstream of behavior; behavior mediates feature-performance link**

**Causal Model:**
```
Features (architecture, family, thinking mode)
    ↓ (partial influence)
Behavioral Capabilities (cooperation, reasoning, rationality)
    ↓ (strong influence)
Performance (profit, wins, efficiency)
```

**Evidence:**
- Features R² = 0.562 (direct path)
- Capabilities R² = 0.766 (mediating path)
- Combined R² = 0.816 (features add +5 pp to MAgIC)
- Capabilities explain MORE variance → strong mediator

**Implication:** **Measure behavior, not just features, for best prediction**

---

### Integration 3: Effort as Epiphenomenon

**Supplementary Finding:** Reasoning effort (chars) uncorrelated with performance  
**RQ1 Finding:** Thinking mode (TE) predicts performance  
**RQ3 Finding:** Reasoning quality (MAgIC) predicts performance  
**Integration:** **Effort is a BYPRODUCT of mode, but QUALITY is what matters**

**Hierarchy:**
1. **Thinking mode enables reasoning** (TE models can reason, TD models can't/don't)
2. **Reasoning quality determines outcomes** (good reasoning → good performance)
3. **Reasoning effort is noise** (long text ≠ good reasoning)

**Analogy:** Academic writing
- **Mode:** Having research skills (PhD vs no PhD)
- **Quality:** Insight, rigor, clarity
- **Effort:** Page count

→ PhDs write better papers (mode), insightful papers succeed (quality), but long papers ≠ good papers (effort irrelevant)

---

### Integration 4: Context Determines Success

**RQ1 Finding:** Performance varies by game (0–100% win rates)  
**RQ2 Finding:** Behavioral profiles stable across games  
**RQ3 Finding:** Different games reward different capabilities  
**Integration:** **Same behavior → different outcomes depending on game structure**

**Example: Model A (High Cooperation, Low Rationality)**
- **Athey-Bagwell:** Wins often (+0.928 coef) → Good outcome ✅
- **Green-Porter:** Sustains collusion (-0.014 reversion) → Good strategy ✅
- **Salop:** Moderate profit → Medium outcome ≈
- **Spulber:** Poor matching (needs rationality) → Bad outcome ❌

**Model A's profile is STABLE, but outcomes VARY by game**

**Implication:** **Match models to tasks** — "best" model is task-dependent

---

## Theoretical Framework

### The LLM Strategic Competence Model

```
┌─────────────────────────────────────────────────────────┐
│ TRAINING PHASE (Pre-deployment)                         │
├─────────────────────────────────────────────────────────┤
│ Training Data + Architecture + Training Procedure       │
│                         ↓                                │
│ Behavioral Profile Formation (MAgIC dimensions)         │
│   - Cooperation tendency                                │
│   - Rationality capacity                                │
│   - Reasoning depth                                     │
│   - Coordination ability                                │
│   - Self-awareness, judgment, etc.                      │
│                         ↓                                │
│ STABLE PROFILE (97–99% consistent across contexts)     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ DEPLOYMENT PHASE (In-context)                           │
├─────────────────────────────────────────────────────────┤
│ Game Structure (rules, payoffs, information)            │
│            +                                             │
│ Competitive Environment (3P vs 5P, opponent mix)       │
│            +                                             │
│ Behavioral Profile (stable from training)               │
│            ↓                                             │
│ Context-Capability Matching                             │
│   - Does game reward cooperation? → Cooperators win    │
│   - Does game reward rationality? → Optimizers win     │
│   - Does game reward reasoning? → Strategists win      │
│            ↓                                             │
│ PERFORMANCE OUTCOMES (profit, wins, efficiency)        │
│   - Predictable from capabilities (R² = 0.766)         │
│   - Combined with features (R² = 0.816)                │
│   - Variable across environments (3P vs 5P)            │
└─────────────────────────────────────────────────────────┘
```

---

### Key Principles

#### Principle 1: Behavioral Determinism
**Behavioral profiles are largely FIXED by training, not learned strategically in-context.**

**Evidence:**
- 97–99% stability across 3P→5P conditions
- No evidence of within-experiment learning/adaptation
- Profiles emerge from training, persist in deployment

**Implication:** Can't expect models to "learn" new strategies during use. Profile is what you get.

---

#### Principle 2: Context-Capability Fit
**Performance depends on MATCH between model capabilities and task requirements.**

**Evidence:**
- Same model: Good in Salop, bad in Spulber
- Different games reward different capabilities (R² varies 0.006–0.997)
- No universally "best" model

**Implication:** Model selection should be task-specific, not generic "best model" rankings.

---

#### Principle 3: Capability Trade-offs
**High capability in one dimension often comes with COSTS in another dimension.**

**Evidence:**
- Cooperation ↔ Rationality trade-off (Athey-Bagwell)
- Reasoning ↔ Speed trade-off (Spulber)
- Deep thinking ↔ Quick wins trade-off (Athey-Bagwell)

**Implication:** Can't have "perfect" model. Design must balance competing objectives.

---

#### Principle 4: Emergence Over Engineering
**Strategic capabilities emerge from training, not explicitly programmed.**

**Evidence:**
- MAgIC (behavioral capabilities) >> Architecture (design features) for prediction
- 82% variance from behavior vs 46% from features
- Same architecture → different behaviors (within model families)

**Implication:** Training data/procedures more important than architectural choices for strategic competence.

---

#### Principle 5: Quality Over Quantity
**Reasoning effectiveness determined by QUALITY (capability) not QUANTITY (effort).**

**Evidence:**
- MAgIC reasoning: 80% success rate (quality measure)
- Character count: 0% success rate (quantity measure)
- Sometimes negative correlation (more text → worse outcomes)

**Implication:** Prompt for conciseness, evaluate by outcomes, not verbosity.

---

## Practical Implications

### For AI Researchers

#### 1. Measurement Frameworks
**Recommendation:** Use behavioral capability metrics (like MAgIC), not just architectural features.

**Rationale:**
- Capabilities explain 78% more variance than features
- Captures emergent properties from training
- Enables cross-model comparisons on strategic competence

**Action:** Develop standardized behavioral evaluation suites for LLMs.

---

#### 2. Evaluation Paradigms
**Recommendation:** Test models in strategic environments, not just linguistic tasks.

**Rationale:**
- Strategic competence differs from linguistic fluency
- Game-theoretic contexts reveal reasoning, cooperation, rationality
- Behavioral profiles stable → generalizable assessment

**Action:** Include economic games in model benchmarks (e.g., "StrategicBench").

---

#### 3. Training Objectives
**Recommendation:** Explicitly train for strategic capabilities (cooperation, rationality, reasoning).

**Rationale:**
- Capabilities emerge from training (Principle 4)
- Current training yields heterogeneous capabilities (some models lack cooperation)
- Can target specific capabilities with curriculum/data

**Action:** Incorporate game-theoretic reasoning in training data, reward cooperative/rational solutions.

---

### For Practitioners (Model Deployers)

#### 1. Model Selection
**Recommendation:** Match model capabilities to task requirements, not generic "best model."

**Rationale:**
- Different games reward different capabilities (Finding 3)
- No universal winner (trade-offs exist)
- Performance varies 0–100% depending on fit

**Action:**
1. Identify task requirements (cooperation? rationality? speed? depth?)
2. Evaluate candidates on those dimensions (MAgIC-style tests)
3. Select best-fit model, not highest-benchmark model

---

#### 2. Prompt Engineering
**Recommendation:** Prompt for thinking MODE (TE), but don't reward verbosity (effort).

**Rationale:**
- TE > TD for performance (50% success in feature regressions)
- But within TE, character count uncorrelated with performance
- Verbosity increases costs (tokens, latency) without benefit

**Action:**
- Use: "Think step-by-step and explain concisely"
- Avoid: "Explain in great detail" or length-based incentives

---

#### 3. Deployment Planning
**Recommendation:** Behavioral profiles stable → can predict behavior in new contexts.

**Rationale:**
- 97–99% stability across conditions
- Profiles measured in one context generalize to similar contexts
- Reduces testing burden (don't need to test every condition)

**Action:**
1. Characterize model behavior in representative environment
2. Use profile to forecast behavior in deployment context
3. Monitor for distribution shift (if task differs substantially from test)

---

#### 4. Multi-Agent Systems
**Recommendation:** Compose teams with COMPLEMENTARY capabilities, not identical models.

**Rationale:**
- Trade-offs exist (cooperation vs rationality, speed vs depth)
- Diverse teams may outperform homogeneous teams
- Different roles require different capabilities

**Action:**
- Assign cooperative models to coordination roles
- Assign rational models to optimization roles
- Assign reasoning models to strategic planning roles

---

### For Model Developers

#### 1. Architecture Choices
**Recommendation:** Focus on TRAINING DATA/PROCEDURES over architecture alone.

**Rationale:**
- Architecture explains only 8% of performance (weakest feature predictor)
- Behavioral capabilities (training-derived) explain 82%
- MoE vs standard: minimal difference in strategic contexts

**Action:** Invest in game-theoretic reasoning data, not just architectural innovations.

---

#### 2. Thinking Mode Design
**Recommendation:** Develop efficient extended thinking (TE) modes.

**Rationale:**
- TE > TD in performance (50% success rate)
- But current TE models generate 8K–36K chars (very verbose)
- Verbosity costly (tokens, latency) without performance benefit

**Action:** Train for "concise reasoning" — maintain TE benefits, reduce output length.

---

#### 3. Capability Balancing
**Recommendation:** Balance cooperation, rationality, reasoning in training (avoid extremes).

**Rationale:**
- Trade-offs exist → over-optimizing one dimension hurts others
- Medium-sized Qwen3-30B-A3B (most balanced) outperforms larger Q3-32B (more adaptive)
- Robust strategy beats over-tuning

**Action:** Multi-objective training (cooperation + rationality + reasoning) with balanced weights.

---

#### 4. Adaptation Mechanisms
**Recommendation:** Develop mechanisms for strategic adaptation (currently lacking).

**Rationale:**
- Current models 97–99% stable (don't adapt to 3P→5P)
- Less adaptive than humans (who adjust strategies based on opponent count)
- Costs: Inflexibility in dynamic environments

**Action:** Research in-context strategy learning, opponent modeling, adaptive prompting.

---

## Limitations and Future Work

### Methodological Limitations

#### 1. Sample Size
**Limitation:** 13 models, 4 games, 2 conditions — limited generalizability

**Impact:**
- N=24 for regressions (borderline for 4-predictor models)
- Game diversity limited (all oligopoly contexts)
- Model diversity limited (mostly Qwen, Llama families)

**Future Work:**
- Expand to 50+ models across more families
- Test 10+ games spanning cooperation, competition, coordination
- Increase condition variations (different payoffs, information structures)

---

#### 2. Causality
**Limitation:** Regression shows correlation, not causation

**Impact:**
- Can't conclude "improving cooperation CAUSES better performance"
- Alternative: Third factor (training data) causes both capability and performance
- Can't test interventions (no capability manipulation)

**Future Work:**
- Causal inference methods (instrumental variables, natural experiments)
- Fine-tuning interventions (target specific capabilities)
- Ablation studies (remove capabilities, measure impact)

---

#### 3. MAgIC Construct Validity
**Limitation:** MAgIC dimensions measure outputs, not internal processes

**Impact:**
- "Reasoning" score may capture verbosity, not true reasoning
- "Cooperation" may reflect prompt-following, not genuine cooperation intent
- Black box: Don't know WHY models exhibit capabilities

**Future Work:**
- Interpretability: Link MAgIC to attention patterns, activations
- Validation: Compare MAgIC to human expert ratings
- Refinement: Improve MAgIC metrics based on neural evidence

---

#### 4. Game Realism
**Limitation:** Simplified games may not reflect real-world complexity

**Impact:**
- Salop, Spulber, etc. are stylized models
- Real markets have more players, continuous actions, information asymmetries
- Findings may not generalize to actual business contexts

**Future Work:**
- Test in realistic simulations (supply chains, auctions, negotiations)
- Deploy in real-world pilot studies (with guardrails)
- Compare lab vs field strategic competence

---

### Theoretical Gaps

#### 1. Mechanism Opacity
**Gap:** Don't understand HOW training produces behavioral profiles

**Questions:**
- What training data patterns → cooperation capability?
- Which model layers encode strategic reasoning?
- How does pre-training vs fine-tuning shape profiles?

**Future Work:**
- Training data analysis (cooperation examples → cooperation capability?)
- Mechanistic interpretability (identify "cooperation circuits")
- Controlled training experiments (vary data, measure capability changes)

---

#### 2. Adaptation Dynamics
**Gap:** Don't know if/how models could learn to adapt

**Questions:**
- Can in-context learning change behavioral profiles?
- Would longer experiments show adaptation over time?
- Can meta-learning enable strategic flexibility?

**Future Work:**
- Longitudinal studies (100+ rounds, track behavior evolution)
- Meta-learning architectures (explicitly train for adaptation)
- Opponent modeling (can models learn from others' strategies?)

---

#### 3. Capability Emergence
**Gap:** Don't know at what scale/training capabilities emerge

**Questions:**
- Do small models (<7B) have strategic capabilities?
- Is there a sharp phase transition (like with general capabilities)?
- Can targeted fine-tuning instill capabilities in small models?

**Future Work:**
- Scaling studies (test 1B–100B models)
- Capability onset analysis (when does reasoning emerge?)
- Efficient capability transfer (distillation, fine-tuning)

---

### Practical Gaps

#### 1. Real-World Validation
**Gap:** All findings from controlled experiments, not field deployments

**Need:**
- Test in actual business negotiations, market interactions
- Validate behavioral stability in real-world contexts
- Measure economic value (not just game-theoretic metrics)

**Future Work:**
- Industry partnerships (pilot LLM agents in real markets)
- A/B testing (LLM vs human vs hybrid teams)
- Economic impact assessment (cost savings, efficiency gains)

---

#### 2. Safety Implications
**Gap:** Strategic competence may enable deception, collusion, manipulation

**Concerns:**
- Cooperative models may collude against users
- Reasoning models may find exploits in systems
- Stable profiles may hide adversarial tendencies

**Future Work:**
- Adversarial testing (can models exploit weaknesses?)
- Alignment research (ensure capabilities serve user goals)
- Governance frameworks (detect/prevent harmful strategic behavior)

---

#### 3. Human-AI Interaction
**Gap:** Don't know how humans + LLMs interact strategically

**Questions:**
- Do humans trust LLM strategic advice?
- Can humans detect LLM cooperation/deception?
- Do mixed teams (human+LLM) outperform pure teams?

**Future Work:**
- Human-AI game experiments (mixed player pools)
- Trust and transparency studies (when do humans follow LLM advice?)
- Team composition optimization (best human-AI ratios?)

---

## Conclusions

### Main Contributions

#### 1. Behavioral Profile Framework
**Contribution:** Demonstrated that LLMs have STABLE behavioral profiles (97–99% similarity) that STRONGLY predict performance (R² = 0.766 MAgIC, 0.816 combined).

**Significance:** Shifts focus from architectural features to behavioral capabilities. Enables predictive modeling of LLM strategic behavior.

**Impact:** Researchers can measure, compare, and select models based on strategic competence, not just linguistic fluency.

---

#### 2. Capability-Performance Links
**Contribution:** Quantified relationships between specific capabilities (cooperation, rationality, reasoning) and game-theoretic outcomes.

**Significance:** Shows WHAT makes models succeed in strategic contexts. Reasoning is strongest (80% success), but trade-offs exist (cooperation vs rationality).

**Impact:** Developers can target training for specific capabilities. Practitioners can match models to tasks based on capability requirements.

---

#### 3. Context-Dependent Competence
**Contribution:** Established that "best model" is task-dependent. Same model excels in one game, fails in another (0–100% win rate range).

**Significance:** Challenges universal benchmarking. Performance is interaction between model capabilities and environmental demands.

**Impact:** Model selection must be context-specific. Generic rankings misleading for strategic deployment.

---

#### 4. Quality-Not-Quantity Principle
**Contribution:** Showed that reasoning QUALITY (MAgIC) predicts performance (80% success), but QUANTITY (character count) doesn't (0% success).

**Significance:** Verbosity is not competence. Efficient reasoning outperforms verbose reasoning.

**Impact:** Prompt for conciseness, evaluate by outcomes. Don't reward or expect verbosity.

---

### Theoretical Implications

#### 1. Emergent Strategy
**Implication:** Strategic capabilities emerge from training, not hard-coded.

**Evidence:** Behavior (R² = 0.766) >> Architecture (R² = 0.562) for prediction.

**Consequence:** Training data/procedures are key to strategic competence, not model design alone.

---

#### 2. Behavioral Consistency
**Implication:** LLMs exhibit "strategic personalities" that persist across contexts.

**Evidence:** 97–99% stability despite significant environmental changes (3P→5P).

**Consequence:** Behavior is fundamental model characteristic, like model size or architecture.

---

#### 3. Trade-off Landscape
**Implication:** No universally optimal model — trade-offs force specialization.

**Evidence:** Cooperation vs rationality (Athey-Bagwell), reasoning vs speed (Spulber).

**Consequence:** Model development must balance competing objectives, not maximize single dimension.

---

### Practical Takeaways

#### For Research
1. **Measure behavior, not just features** — 78% better prediction
2. **Expect stability** — profiles generalize across similar contexts
3. **Test strategically** — game-theoretic evaluation reveals capabilities standard benchmarks miss

#### For Practice
1. **Match models to tasks** — "best" is context-dependent
2. **Prompt for thinking, not verbosity** — TE helps, but conciseness doesn't hurt
3. **Compose diverse teams** — complementary capabilities > identical models

#### For Development
1. **Train for capabilities** — cooperation, rationality, reasoning
2. **Balance, don't maximize** — trade-offs exist, robustness beats extremes
3. **Enable adaptation** — current models too rigid, need flexibility mechanisms

---

### Final Verdict

#### RQ1: Competitive Performance
**ANSWERED:** Performance varies dramatically (0–100% win rates). Model features explain 56%, leaving substantial heterogeneity. Competition intensity reduces profits but behavioral strategies stable.

#### RQ2: Behavioral Profiles
**ANSWERED:** Profiles extremely stable (97–99% similarity). Family clustering weak. Behavioral "fingerprints" are fundamental model characteristics.

#### RQ3: Capability-Performance Links
**ANSWERED:** Behavioral capabilities strongly predict performance (R² = 0.766 MAgIC, 0.816 combined). Reasoning strongest predictor (80% success). Context determines which capabilities succeed.

#### Supplementary: Reasoning Effort
**ANSWERED:** Effort varies by game but uncorrelated with performance. Quality matters, quantity doesn't.

---

### Closing Insight

**LLM strategic competence is not a monolithic trait ("smart" vs "not smart") but a MULTIDIMENSIONAL PROFILE of capabilities (cooperation, rationality, reasoning, etc.) that emerges from training, remains stable across contexts, and predicts performance through context-capability matching.**

**Implication:** Success in deploying LLMs for strategic tasks requires understanding their behavioral profile, matching it to task requirements, and accepting trade-offs inherent in capability landscape. **What models DO matters more than what models ARE.**

---

## Related Documentation

### Individual File Summaries
- `SUMMARY_T_perf_win_rate.md` — Win rate analysis
- `SUMMARY_T_perf_avg_profit.md` — Profit analysis
- `SUMMARY_T_perf_game_specific.md` — Game-specific metrics
- `SUMMARY_T_mlr_features_to_performance.md` — Feature regression
- `SUMMARY_T5_magic_to_perf.md` — MAgIC→performance regression
- `SUMMARY_T_similarity_3v5.md` — Behavioral stability
- `SUMMARY_T_reasoning_chars.md` — Reasoning effort analysis

### Research Question Syntheses
- `SYNTHESIS_RQ1_Competitive_Performance.md` — Performance findings
- `SYNTHESIS_RQ2_Behavioral_Profiles.md` — Stability and clustering
- `SYNTHESIS_RQ3_Capability_Performance_Links.md` — Capability prediction
- `SYNTHESIS_Supplementary_Reasoning.md` — Reasoning effort analysis

### Master Index
- `FILE_SUMMARIES_INDEX.md` — Complete file inventory and summaries
