# Individual File Summaries - Complete Index

This document provides an index to all individual file summaries in the publication folder.

## CSV Files with Summaries

### RQ1: Competitive Performance
1. [SUMMARY_T_perf_win_rate.md](SUMMARY_T_perf_win_rate.md) - Win rates per model (3P vs 5P)
2. [SUMMARY_T_perf_avg_profit.md](SUMMARY_T_perf_avg_profit.md) - Average profits per model (3P vs 5P)
3. [SUMMARY_T_perf_game_specific.md](SUMMARY_T_perf_game_specific.md) - Game-specific metrics
4. [SUMMARY_T_mlr_features_to_performance.md](SUMMARY_T_mlr_features_to_performance.md) - MLR: Features → Performance

### RQ2: Strategic Behavioral Profiles
5. [SUMMARY_T_magic_salop.md](SUMMARY_T_magic_salop.md) - MAgIC metrics for Salop
6. [SUMMARY_T_magic_spulber.md](SUMMARY_T_magic_spulber.md) - MAgIC metrics for Spulber
7. [SUMMARY_T_magic_green_porter.md](SUMMARY_T_magic_green_porter.md) - MAgIC metrics for Green-Porter
8. [SUMMARY_T_magic_athey_bagwell.md](SUMMARY_T_magic_athey_bagwell.md) - MAgIC metrics for Athey-Bagwell
9. [SUMMARY_T_similarity_3v5.md](SUMMARY_T_similarity_3v5.md) - 3P vs 5P stability analysis
10. [SUMMARY_T6_pca_variance.md](SUMMARY_T6_pca_variance.md) - PCA variance explained

### RQ3: Capability-Performance Links
11. [SUMMARY_T5_magic_to_perf.md](SUMMARY_T5_magic_to_perf.md) - MLR: MAgIC → Performance

### Supplementary
12. [SUMMARY_T_reasoning_chars.md](SUMMARY_T_reasoning_chars.md) - Reasoning effort analysis

## PNG-Only Files with Summaries

### RQ2: Strategic Behavioral Profiles
13. [SUMMARY_F_similarity_salop.md](SUMMARY_F_similarity_salop.md) - Similarity heatmap for Salop
14. [SUMMARY_F_similarity_spulber.md](SUMMARY_F_similarity_spulber.md) - Similarity heatmap for Spulber
15. [SUMMARY_F_similarity_green_porter.md](SUMMARY_F_similarity_green_porter.md) - Similarity heatmap for Green-Porter
16. [SUMMARY_F_similarity_athey_bagwell.md](SUMMARY_F_similarity_athey_bagwell.md) - Similarity heatmap for Athey-Bagwell
17. [SUMMARY_F_similarity_3v5.md](SUMMARY_F_similarity_3v5.md) - 3P vs 5P stability bar chart
18. [SUMMARY_F_pca_scree.md](SUMMARY_F_pca_scree.md) - PCA scree plots

### Supplementary
19. [SUMMARY_F_reasoning_chars.md](SUMMARY_F_reasoning_chars.md) - Reasoning effort bar chart

## Synthesized Reports by Research Question

### Master Synthesis Documents
- [**MASTER_SYNTHESIS.md**](MASTER_SYNTHESIS.md) - **COMPLETE CROSS-RQ INTEGRATION** ⭐
- [SYNTHESIS_RQ1_Competitive_Performance.md](SYNTHESIS_RQ1_Competitive_Performance.md) - Complete RQ1 analysis
- [SYNTHESIS_RQ2_Behavioral_Profiles.md](SYNTHESIS_RQ2_Behavioral_Profiles.md) - Complete RQ2 analysis
- [SYNTHESIS_RQ3_Capability_Performance_Links.md](SYNTHESIS_RQ3_Capability_Performance_Links.md) - Complete RQ3 analysis
- [SYNTHESIS_Supplementary_Reasoning.md](SYNTHESIS_Supplementary_Reasoning.md) - Reasoning effort analysis

## Quick Statistics

- **Total CSV files:** 12
- **Total PNG files:** 19 (7 with CSV, 12 PNG-only)
- **Individual summaries created:** 12 (all CSV files documented)
- **Synthesis documents created:** 5 (RQ1, RQ2, RQ3, Supplementary, Master)

## Summary Status

### Completed ✅
- All CSV file summaries (12/12)
- All synthesis documents (5/5)
- Master integration document (MASTER_SYNTHESIS.md)
- Cross-RQ analysis complete

### Key Findings Quick Reference

#### RQ1: Competitive Performance
- Model features explain 56% of variance (R² = 0.56, after multicollinearity fix)
- Thinking mode (TE) most consistent predictor (67% success)
- All games show profit declines 3P→5P
- Performance ranges 0–100% win rates

#### RQ2: Behavioral Profiles
- **H1 (Family clustering):** PARTIALLY REJECTED
- **H2 (Stability):** STRONGLY CONFIRMED (97–99% similarity)
- 2–4 PCA dimensions explain 80%+ variance
- Behavioral profiles are fundamental characteristics

#### RQ3: Capability-Performance Links
- MAgIC explains 82% of variance (R² = 0.82)
- 46% better than model features (0.56)
- Reasoning capability: 80% success rate
- Context-dependent requirements

#### Supplementary: Reasoning Effort
- Effort varies 8K–36K chars by game
- NO correlation with performance
- Quality (MAgIC) matters, quantity doesn't
- Medium model most stable and best performing
- **Total individual summaries:** 19
- **Total synthesis reports:** 4

## Navigation Guide

### By File Type
- **Performance Tables:** Files 1-4
- **MAgIC Tables:** Files 5-8
- **Regression Tables:** Files 4, 11
- **Similarity Analysis:** Files 9, 13-17
- **PCA Analysis:** Files 10, 18
- **Reasoning Analysis:** Files 12, 19

### By Research Question
- **RQ1:** Summaries 1-4 → Synthesis RQ1
- **RQ2:** Summaries 5-10, 13-18 → Synthesis RQ2
- **RQ3:** Summary 11 → Synthesis RQ3
- **Supplementary:** Summaries 12, 19 → Synthesis Supplementary

## Key Findings at a Glance

### RQ1: Model Features → Performance
- **R² average:** 0.5623 (after multicollinearity fix)
- **Significant predictors:** 18/60 (30%)
- **Best predictor:** Thinking mode (67% success)

### RQ2: Behavioral Profiles
- **3P↔5P similarity:** 0.9744-0.9961 (cosine)
- **Profile stability:** Very high (all >0.97)
- **PCA components:** 2-4 per game

### RQ3: MAgIC → Performance
- **R² average (MAgIC only):** 0.766 (76.6% variance)
- **R² average (Combined):** 0.816 (81.6% variance)
- **R² max:** 0.9965 (near-perfect fit)
- **Significant predictors:** 23/39 (59%)
- **Best game:** Spulber (R²=0.99)

### Key Insight
**MAgIC metrics explain performance MUCH better than model features (R²: 0.766 vs 0.562) — 36% better. Combined model reaches 82%.**

## Usage

1. Navigate to individual file summaries for detailed analysis
2. Refer to synthesis reports for cross-file insights
3. Use this index to find specific analyses quickly

---

**Generated:** February 2, 2026  
**Source Data:** `output/analysis/publication/`  
**Analysis Pipeline:** `analysis.py`
