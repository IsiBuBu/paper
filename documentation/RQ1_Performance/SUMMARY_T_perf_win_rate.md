# Summary: Win Rate Performance Analysis

**Table:** `T_perf_win_rate.csv` | **Research Question:** RQ1 - Competitive Performance  
**Models:** 13 (12 LLMs + Random) | **Games:** 4 | **Structures:** 3P/5P

---

## Data Overview

Win rates across four oligopoly games under two market structures (3-player baseline vs 5-player expansion). Each cell contains mean ± std with p-values testing 3P→5P structural sensitivity.

**Games:** Athey-Bagwell (entry deterrence), Green-Porter (collusion), Salop (spatial competition), Spulber (mechanism design)

---

## Overall Performance Hierarchy

### Top Performers (Cross-Game Average, 3P)
1. **Q3-14B (TE)**: 87.5% — Highest overall competence
2. **Q3-32B (TE)**: 80.0% — Consistent strong performance  
3. **Q3-235B Inst**: 73.8% — Large-model robustness
4. **Qwen3-30B-A3B (TE)**: 72.5% — Thinking-enhanced advantage
5. **Q3-32B (TD)**: 71.3% — Default thinking strength

### Bottom Performers
- **Random**: 12.5% (baseline)
- **L3.1-8B**: 13.8% — Size constraint evident
- **L3.3-70B**: 48.8% — Game-selective failure (0% in Salop/Spulber)
- **L4-Scout**: 55.0% — Inconsistent (100% in 2 games, 0% in 2 games)

---

## Game-Specific Patterns

### Athey-Bagwell (Entry Deterrence)
- **Highest win rates overall** — Most models >85%
- **Structural sensitivity**: **p=0.0119*** — Only game showing significant 3P→5P effect
- **3P leaders**: L4-Scout (100%), Q3-14B TE (99.2%), Q3-235B (99.2%)
- **5P shift**: 7 models achieve perfect 100% win rate in 5P (vs 1 in 3P)
- **Interpretation**: More players simplify entry deterrence strategies

### Green-Porter (Collusion Sustainability)
- **Easiest game** — Most models >90%, several at 100%
- **Perfect structural stability**: **p=1.000 ns** — No 3P→5P effect
- **3P/5P leaders**: Multiple models maintain 100% across conditions
- **Consistent excellence**: Q3-14B (TD), Q3-235B, Q3-32B (TE), L4-Scout
- **Interpretation**: Collusion detection robust to market structure

### Salop (Spatial Competition)
- **Hardest game** — Highest variance (0–80%), lowest average (~40%)
- **No structural effect**: **p=0.2087 ns**
- **3P leaders**: Qwen3-30B-A3B (TE) and Q3-32B (TE) at 80%
- **Complete failures**: L3.3-70B, L4-Maverick, L4-Scout, Random (all 0%)
- **Interpretation**: Spatial reasoning separates capable from incapable models

### Spulber (Mechanism Design)
- **Moderate difficulty** — High spread (0–92%)
- **No structural effect**: **p=0.1497 ns**
- **3P leaders**: Qwen3-30B-A3B (TD) at 92%
- **Strategic variance**: Clear capability hierarchy, no mid-range performers
- **Interpretation**: Mechanism understanding is binary (success/failure)

---

## Structural Sensitivity Analysis

| Game | 3P→5P Change | P-Value | Effect |
|------|-------------|---------|--------|
| **Athey-Bagwell** | +7.8% | **0.0119*** | ✅ Significant |
| Green-Porter | -0.2% | 1.000 ns | ❌ None |
| Salop | +5.9% | 0.2087 ns | ❌ None |
| Spulber | +0.6% | 0.1497 ns | ❌ None |

**Key Finding**: **Only 1/4 games** shows structural sensitivity. Game mechanics dominate over market size effects.

---

## Model Capability Insights

### Thinking Enhancement Effect (TD vs TE)
- **Q3-14B**: TE +37.5% over TD (87.5% vs 50.0%)
- **Qwen3-30B-A3B**: TE +7.5% over TD (72.5% vs 65.0%)
- **Q3-32B**: TE +8.8% over TD (80.0% vs 71.3%)
- **Pattern**: Thinking enhancement benefits vary by base model capability

### Size vs Architecture
- **Large models not always better**: Q3-235B (73.8%) < Q3-14B TE (87.5%)
- **Small models struggle**: L3.1-8B (13.8%) barely above random
- **Architecture matters**: Qwen3-30B outperforms some 70B models

---

## Key Takeaways

1. **Performance hierarchy is robust** — Q3-14B (TE) dominates across 3 games
2. **Game difficulty order**: Green-Porter (easiest) > Athey-Bagwell > Spulber > Salop (hardest)
3. **Structural effects are game-specific** — Only entry deterrence (Athey-Bagwell) sensitive to 3P→5P
4. **Model gaps are large** — 87.5% (best) vs 12.5% (random) vs 13.8% (worst LLM)
5. **Game-selective competence exists** — Some models excel in 2 games, fail completely in 2 others
2. **Game-specific specialization:** Some models excel in certain games but fail in others
3. **Structural stability varies:** Only Athey-Bagwell shows significant sensitivity to market structure
4. **Thinking vs. No-thinking unclear:** Both TD (thinking disabled) and TE (thinking enabled) models appear in top performers

## Statistical Notes

- **Format:** Values shown as `mean ± std ***` where stars indicate 3P vs 5P significance
- **P-values:** Paired t-test comparing 3P vs 5P performance across models
- **Significance levels:** * p<0.05, ** p<0.01, *** p<0.001
- **Missing data:** Some combinations show N/A (typically Random agent)

## Next Steps for Analysis

1. Compare with `T_perf_avg_profit.csv` to see if win rate correlates with profit
2. Check `T_mlr_features_to_performance.csv` to identify which model features predict win rate
3. Examine `T_magic_{game}.csv` files to understand behavioral differences between high/low performers

---

**File Location:** `output/analysis/publication/T_perf_win_rate.csv`  
**Corresponding Figure:** `T_perf_win_rate.png`  
**Data Format:** CSV with 13 rows × 13 columns
