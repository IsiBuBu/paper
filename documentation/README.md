# 📚 Documentation Index

**Complete Analysis Documentation for LLM Strategic Behavior Research**

---

## 🎯 Quick Start

### **START HERE** → [`Master/MASTER_SYNTHESIS.md`](Master/MASTER_SYNTHESIS.md)
Complete integrated analysis across all research questions

### By Research Question
- **RQ1: Performance** → [`RQ1_Performance/`](RQ1_Performance/)
- **RQ2: Profiles** → [`RQ2_Profiles/`](RQ2_Profiles/)
- **RQ3: Capabilities** → [`RQ3_Capabilities/`](RQ3_Capabilities/)
- **Supplementary** → [`Supplementary/`](Supplementary/)

---

## 📁 Folder Structure

```
documentation/
├── Master/                          # Cross-RQ integration
│   ├── MASTER_SYNTHESIS.md         ⭐ Complete synthesis
│   ├── DOCUMENTATION_COMPLETE.md    Entry point
│   └── FILE_SUMMARIES_INDEX.md      File inventory
│
├── RQ1_Performance/                 # Competitive performance
│   ├── SYNTHESIS_RQ1_Competitive_Performance.md
│   ├── SUMMARY_T_perf_win_rate.md
│   ├── SUMMARY_T_perf_avg_profit.md
│   ├── SUMMARY_T_perf_game_specific.md
│   └── SUMMARY_T_mlr_features_to_performance.md
│
├── RQ2_Profiles/                    # Behavioral profiles
│   ├── SYNTHESIS_RQ2_Behavioral_Profiles.md
│   ├── SUMMARY_T_magic_salop.md
│   ├── SUMMARY_T_magic_spulber.md (to create)
│   ├── SUMMARY_T_magic_green_porter.md (to create)
│   ├── SUMMARY_T_magic_athey_bagwell.md (to create)
│   ├── SUMMARY_T_similarity_3v5.md
│   ├── SUMMARY_T6_pca_variance.md (to create)
│   ├── SUMMARY_F_similarity_salop.md (to create)
│   ├── SUMMARY_F_similarity_spulber.md (to create)
│   ├── SUMMARY_F_similarity_green_porter.md (to create)
│   ├── SUMMARY_F_similarity_athey_bagwell.md (to create)
│   └── SUMMARY_F_pca_scree.md (to create)
│
├── RQ3_Capabilities/                # Capability-performance links
│   ├── SYNTHESIS_RQ3_Capability_Performance_Links.md
│   └── SUMMARY_T5_magic_to_perf.md
│
└── Supplementary/                   # Reasoning effort
    ├── SYNTHESIS_Supplementary_Reasoning.md
    ├── SUMMARY_T_reasoning_chars.md
    ├── SUMMARY_F_reasoning_chars.md (to create)
    └── SUMMARY_F_similarity_3v5.md (to create)
```

---

## 📊 Documentation Status

### ✅ Complete (12 files)
- [x] All CSV summaries for RQ1 (4 files)
- [x] Core RQ2 summaries (2 files)
- [x] RQ3 summary (1 file)
- [x] Supplementary summary (1 file)
- [x] All synthesis documents (4 files)
- [x] Master synthesis (1 file)

### 🚧 To Create (9 files)
- [ ] T_magic_{game}.md for 3 remaining games
- [ ] T6_pca_variance.md
- [ ] F_similarity_{game}.md for 4 games
- [ ] F_pca_scree.md
- [ ] F_reasoning_chars.md
- [ ] F_similarity_3v5.md

---

## 🔑 Key Findings Quick Reference

### Main Result
**Behavioral capabilities (MAgIC) explain 77% of performance variance** — 36% better than architectural features (56%). Combined models reach 82%.

### By Research Question

| RQ | Finding | Evidence | Status |
|----|---------|----------|--------|
| **RQ1** | Features explain 46%; thinking mode strongest | Win rates, profits, MLR | ✅ Complete |
| **RQ2** | Profiles 97–99% stable; weak clustering | Similarity, PCA | ✅ Core complete |
| **RQ3** | MAgIC explains 82%; reasoning 80% success | MAgIC→perf MLR | ✅ Complete |
| **Supp** | Effort uncorrelated; quality matters | Reasoning chars | ✅ Complete |

---

## 📈 Statistical Highlights

### Model Fit Comparison
- **MAgIC → Performance:** R² = 0.766 (76.6% variance explained)
- **Combined (MAgIC + Features):** R² = 0.816 (81.6% variance explained)
- **Features → Performance:** R² = 0.562 (56.2% variance explained)
- **MAgIC Advantage:** **36% better** than features alone

### Behavioral Stability
- **Athey-Bagwell:** 99.6% similarity (3P vs 5P)
- **Spulber:** 99.3% similarity
- **Green-Porter:** 98.2% similarity
- **Salop:** 97.4% similarity

### Capability Success Rates
- **Reasoning:** 80% (8/10 tests significant)
- **Rationality:** 67% (6/9 tests significant)
- **Cooperation:** 63% (5/8 tests significant)

---

## 🎓 How to Use

### For Quick Overview
1. Read [`Master/DOCUMENTATION_COMPLETE.md`](Master/DOCUMENTATION_COMPLETE.md)
2. Scan key findings above
3. Review RQ synthesis summaries

### For Detailed Analysis
1. Go to relevant RQ folder
2. Read SYNTHESIS document first
3. Drill into SUMMARY files for specifics

### For Paper Writing
1. Use synthesis documents for Results section
2. Extract tables from summary files
3. Reference specific findings with file paths
4. Use practical implications for Discussion

---

## 📝 File Naming Convention

- `SUMMARY_*.md` — Individual file/figure summaries
- `SYNTHESIS_*.md` — Research question syntheses
- `MASTER_*.md` — Cross-RQ integration
- `T_*.csv` — Table data files
- `F_*.png` — Figure image files

---

## 🔗 Navigation

### From Any Summary → Synthesis
- Each summary links to relevant synthesis document
- Example: `SUMMARY_T_perf_win_rate.md` → `SYNTHESIS_RQ1_Competitive_Performance.md`

### From Synthesis → Master
- Each synthesis links to master integration
- Example: `SYNTHESIS_RQ1_*.md` → `MASTER_SYNTHESIS.md`

### Cross-References
- Related files listed at end of each document
- Follow links to explore connections

---

## 📧 Contact

**Documentation Version:** 1.5 (Organized)
**Last Updated:** February 2, 2026  
**Status:** Core complete, PNG summaries in progress

---

**🎯 Start with [`Master/MASTER_SYNTHESIS.md`](Master/MASTER_SYNTHESIS.md) for complete picture!**
