# Analysis Module Documentation

## Overview

The `analysis/` directory contains modular components for analyzing LLM behavior in economic games. The pipeline processes raw experiment data to generate publication-ready tables and figures.

## Architecture

```
analysis/
├── __init__.py
├── data_loader.py          # Load performance & MAgIC metrics
├── feature_extractor.py    # Extract model features from names
├── utils.py                # Shared utilities & constants
├── table_generator.py      # Generate publication tables
├── figure_generator.py     # Generate publication figures
└── engine/
    ├── analyze_metrics.py      # Compute metrics from raw data
    └── create_summary_csvs.py  # Create summary CSVs
```

## Quick Start

### Run Full Pipeline

```bash
python run_analysis.py
```

This executes:
1. **Analyze** raw experiment data → compute metrics
2. **Summarize** metrics → create CSVs
3. **Load** data with/without random agent
4. **Extract** architectural features from model names
5. **Generate** 17 publication tables (CSV + PNG)
6. **Generate** 9 publication figures (PNG)

Output: `output/analysis/publication/`

## Modules

### 1. DataLoader (`data_loader.py`)

Loads and preprocesses experimental data.

```python
from analysis.data_loader import DataLoader

loader = DataLoader(analysis_dir, config_path, experiments_dir)

# Load data
perf, magic = loader.load(include_random=False)
token_df = loader.load_token_data()

# Get metadata
display_name = loader.get_display_name("meta-llama/Llama-3.1-8B-Instruct")
is_thinking = loader.get_thinking_status("meta-llama/Llama-3.3-70B-Instruct")
```

**Features:**
- Loads `performance_metrics.csv` and `magic_behavioral_metrics.csv`
- Filters excluded models (gemma, optionally random_agent)
- Extracts reasoning effort from experiment JSONs
- Provides display names and thinking status

### 2. FeatureExtractor (`feature_extractor.py`)

Extracts architectural features from model names.

```python
from analysis.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(model_configs)
features_df = extractor.extract_features(model_list)
```

**Extracted Features (7):**
1. `architecture_moe` (binary) - Is this a Mixture-of-Experts model?
2. `size_params` (float) - Active parameters in billions
3. `family` (categorical) - Model family (qwen, llama, gpt, etc.)
4. `family_encoded` (int) - Encoded family for regression
5. `family_version` (ordinal) - Within-family generation (0, 1, 2)
6. `thinking` (binary) - Extended reasoning capability
7. `model` (str) - Original model identifier

**Examples:**
- `Qwen/Qwen2.5-72B-Instruct` → size=72, family=qwen, version=0
- `meta-llama/Llama-3.3-70B-Instruct` → size=70, family=llama, version=1
- `meta-llama/Llama-4-Scout-17B-128E` → MoE=1, size=17, family=llama, version=2

### 3. Utils (`utils.py`)

Shared utilities and constants.

```python
from analysis.utils import sig_stars, format_value, remove_collinear_predictors
from analysis.utils import GAME_CONFIGS, METRIC_DIRECTION, MODEL_FEATURES

# Convert p-value to stars
stars = sig_stars(0.001)  # → '***'

# Format statistics
formatted = format_value(0.523, 0.042, '**')  # → "0.523 ± 0.042 **"

# Remove collinear predictors (VIF-based)
remaining = remove_collinear_predictors(df, predictor_list, threshold=0.95)
```

**Constants:**
- `GAME_CONFIGS` - Metrics tracked per game
- `METRIC_DIRECTION` - Whether higher is better (↑) or worse (↓)
- `MODEL_FEATURES` - List of architectural features for regression

### 4. TableGenerator (`table_generator.py`)

Generates publication tables (CSV + PNG).

```python
from analysis.table_generator import TableGenerator

table_gen = TableGenerator(
    perf_with_random,
    magic_with_random,
    perf_no_random,
    magic_no_random,
    features_df,
    output_dir,
    loader
)

# Generate specific table
table_gen.performance_win_rate_table()
table_gen.mlr_features_to_performance()

# Generate all tables
table_gen.generate_all(token_df)
```

**Generated Tables (17 files):**

**RQ1: Competitive Performance**
- `T_perf_win_rate.csv/.png` - Win rates by game/condition
- `T_perf_avg_profit.csv/.png` - Average profits
- `T_perf_game_specific.csv/.png` - Game-specific metrics
- `T_mlr_features_to_performance.csv/.png` - Features → Performance regression

**RQ2: Strategic Behavioral Profiles**
- `T_magic_{game}.csv/.png` (4 games) - MAgIC metrics per game
- `T6_pca_variance.csv/.png` - PCA variance explained

**RQ3: Capability-Performance Links**
- `T5_magic_to_perf.csv/.png` - MAgIC → Performance regression
- `T7_combined_to_perf.csv/.png` - Combined (MAgIC + Features) → Performance

**Supplementary**
- `T_reasoning_chars.csv/.png` - Reasoning effort (thinking models)

**Table Formats:**
- **CSV**: Raw values with separate `p_value` columns for reproducibility
- **PNG**: Formatted with significance stars embedded (* p<.05, ** p<.01, *** p<.001)

### 5. FigureGenerator (`figure_generator.py`)

Generates publication figures (PNG).

```python
from analysis.figure_generator import FigureGenerator

fig_gen = FigureGenerator(
    perf_with_random,
    magic_with_random,
    perf_no_random,
    magic_no_random,
    features_df,
    output_dir,
    loader
)

# Generate specific figure
fig_gen.similarity_matrix('salop')
fig_gen.pca_scree()

# Generate all figures
fig_gen.generate_all(token_df)
```

**Generated Figures (9 files):**

**RQ2: Strategic Behavioral Profiles**
- `F_similarity_{game}.png` (4 games) - Cosine similarity heatmaps
- `F_similarity_3v5.png` - 3P vs 5P stability comparison
- `T_similarity_3v5.csv` - Numerical similarity values
- `F_pca_scree.png` - PCA scree plots (variance explained)

**Supplementary**
- `F_reasoning_chars.png` - Reasoning effort by game/condition

**Figure Properties:**
- Resolution: 300 DPI (publication quality)
- Format: PNG with transparent background
- Annotations: Significance stars embedded
- Style: Seaborn whitegrid

## Research Questions

### RQ1: Competitive Performance
**Hypothesis:** Newer/larger/better LLMs achieve better game performance.

**Independent Variables:** Model features (size, architecture, family, version, thinking)  
**Dependent Variables:** Win rate, average profit, game-specific metrics

**Tables:** T_perf_*, T_mlr_features_to_performance  
**Random Agent:** INCLUDED in performance tables, EXCLUDED from regression

### RQ2: Strategic Behavioral Profiles
**Hypotheses:**
1. Same-family LLMs have similar behavioral profiles
2. Profiles are stable across conditions (3P vs 5P)

**Tables:** T_magic_{game}, T6_pca_variance  
**Figures:** F_similarity_{game}, F_similarity_3v5, F_pca_scree  
**Random Agent:** INCLUDED (should cluster separately)

### RQ3: Capability-Performance Links
**Hypothesis:** Behavioral profiles explain competitive performance.

**Tables:** T5_magic_to_perf (MAgIC → Performance), T7_combined_to_perf (Combined)  
**Random Agent:** EXCLUDED

## Dependencies

### Required (Core)
- `pandas` - Data manipulation
- `numpy` - Numerical operations

### Optional (Analysis)
- `statsmodels` - OLS regression (enables regression tables)
- `scipy` - Statistical tests (enables p-value calculations)
- `sklearn` - PCA, feature encoding (enables PCA tables & figures)
- `matplotlib` - Plotting (enables PNG generation)
- `seaborn` - Statistical visualization (enables heatmaps)

**Graceful Degradation:** If optional dependencies are missing, the pipeline skips those components with warnings.

### Install All Dependencies

```bash
pip install pandas numpy statsmodels scipy scikit-learn matplotlib seaborn
```

## Output Structure

```
output/analysis/
├── performance_metrics.csv       # Raw performance data
├── magic_behavioral_metrics.csv  # Raw MAgIC data
└── publication/                  # Publication-ready outputs
    ├── T_perf_win_rate.csv
    ├── T_perf_win_rate.png
    ├── T_mlr_features_to_performance.csv
    ├── T_mlr_features_to_performance.png
    ├── T5_magic_to_perf.csv
    ├── T5_magic_to_perf.png
    ├── T7_combined_to_perf.csv
    ├── T7_combined_to_perf.png
    ├── F_similarity_salop.png
    └── ...
```

## Extending the Pipeline

### Add a New Table

1. Create method in `TableGenerator`:

```python
def my_new_table(self):
    """Generate my custom table."""
    logger.info("T_my_table")
    
    # Process data
    df = self.perf_df.copy()
    # ... analysis logic ...
    
    # Save CSV
    df.to_csv(self.output_dir / "T_my_table.csv", index=False)
    
    # Save PNG
    self._save_png(df, "T_my_table.png", "My Table Title")
```

2. Add to `generate_all()`:

```python
def generate_all(self, token_df=None):
    # ...existing tables...
    self.my_new_table()
```

### Add a New Figure

1. Create method in `FigureGenerator`:

```python
def my_new_figure(self):
    """Generate my custom figure."""
    logger.info("F_my_figure")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... plotting logic ...
    
    plt.tight_layout()
    plt.savefig(self.output_dir / "F_my_figure.png", dpi=300, bbox_inches='tight')
    plt.close()
```

2. Add to `generate_all()`:

```python
def generate_all(self, token_df=None):
    # ...existing figures...
    self.my_new_figure()
```

### Add a New Model Feature

1. Update `FeatureExtractor.extract_features()`:

```python
def extract_features(self, models):
    for model in models:
        # ...existing extraction...
        new_feature = self._extract_new_feature(model)
        record['new_feature'] = new_feature
```

2. Add to `MODEL_FEATURES` in `utils.py`:

```python
MODEL_FEATURES = [
    'architecture_moe',
    'size_params',
    # ...existing features...
    'new_feature'  # Add here
]
```

## Testing

### Test Individual Modules

```bash
# Test imports
python test_refactored_modules.py

# Test feature extraction
python -c "
from analysis.feature_extractor import FeatureExtractor
extractor = FeatureExtractor({})
features = extractor.extract_features(['meta-llama/Llama-3.1-8B-Instruct'])
print(features)
"
```

### Test Full Pipeline

```bash
# Run complete analysis
python run_analysis.py

# Check outputs
ls -lh output/analysis/publication/
```

## Troubleshooting

### Issue: "statsmodels not available"
**Solution:** Install dependencies: `pip install statsmodels scipy`

### Issue: "sklearn not available"
**Solution:** Install scikit-learn: `pip install scikit-learn`

### Issue: "matplotlib not available"
**Solution:** Install plotting libraries: `pip install matplotlib seaborn`

### Issue: "No data for combined regression"
**Solution:** Ensure both performance and MAgIC data exist, and models are not filtered out

### Issue: VIF calculation fails
**Solution:** This is expected when predictors are constant or perfectly collinear. The pipeline removes such predictors automatically.

## Performance

- **Runtime:** ~2-5 minutes for full pipeline (depends on data size)
- **Memory:** ~500MB peak (for sklearn PCA on large datasets)
- **Output Size:** ~5-10MB (all tables + figures)

## Citation

When using this analysis pipeline, please cite the MAgIC framework paper.

## License

See project LICENSE file.

## Contributors

- Analysis refactoring: February 2026
- Original pipeline: Earlier version
