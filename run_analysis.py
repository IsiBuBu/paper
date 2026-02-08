#!/usr/bin/env python3
"""
MAgIC Analysis Pipeline - Main Entry Point

Usage:
    python run_analysis.py

This script runs the complete analysis pipeline:
1. Analyzes raw experiment data to compute metrics
2. Creates summary CSVs
3. Loads and preprocesses data
4. Generates publication tables (CSV + PNG)
5. Generates publication figures

Output: output/analysis/publication/
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.engine.analyze_metrics import MetricsAnalyzer
from analysis.engine.create_summary_csvs import SummaryCreator
from analysis.data_loader import DataLoader
from analysis.feature_extractor import FeatureExtractor
from config.config import get_experiments_dir, get_analysis_dir

# Optional imports (will be checked before use)
try:
    from analysis.table_generator import TableGenerator
    TABLES_AVAILABLE = True
except ImportError:
    TABLES_AVAILABLE = False
    print("⚠️  TableGenerator not available")

try:
    from analysis.figure_generator import FigureGenerator
    FIGURES_AVAILABLE = True
except ImportError:
    FIGURES_AVAILABLE = False
    print("⚠️  FigureGenerator not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("AnalysisPipeline")


def main():
    """Run the complete MAgIC analysis pipeline."""
    
    logger.info("="*60)
    logger.info("🚀 MAgIC ANALYSIS PIPELINE")
    logger.info("="*60)
    
    # Setup directories
    exp_dir = get_experiments_dir()
    ana_dir = get_analysis_dir()
    ana_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Step 1: Analyze raw experiment data
        logger.info("\n📊 Step 1: Analyzing experiment data...")
        MetricsAnalyzer().analyze_all_games()
        
        # Step 2: Create summary CSVs
        logger.info("\n📝 Step 2: Creating summary CSVs...")
        SummaryCreator().create_all_summaries()
        
        # Step 3: Load data
        logger.info("\n📥 Step 3: Loading data...")
        config_path = Path("config/config.json")
        loader = DataLoader(
            ana_dir,
            config_path if config_path.exists() else None,
            exp_dir
        )
        
        # Load with and without random agent
        perf_with_random, magic_with_random = loader.load(include_random=True)
        perf_no_random, magic_no_random = loader.load(include_random=False)
        
        # Load token/reasoning data
        token_df = loader.load_token_data()
        logger.info(f"   Loaded token data: {len(token_df)} records")
        
        # Step 4: Extract model features
        logger.info("\n🔧 Step 4: Extracting model features...")
        all_models = list(
            set(perf_no_random['model'].unique()) | 
            set(magic_no_random['model'].unique())
        )
        
        extractor = FeatureExtractor(loader.model_configs)
        features_df = extractor.extract_features(all_models)
        logger.info(f"   Extracted features for {len(features_df)} models")
        
        # Step 5: Generate publication tables
        pub_dir = ana_dir / "publication"
        pub_dir.mkdir(exist_ok=True, parents=True)
        
        if TABLES_AVAILABLE:
            logger.info("\n📋 Step 5: Generating publication tables...")
            table_gen = TableGenerator(
                perf_with_random,
                magic_with_random,
                perf_no_random,
                magic_no_random,
                features_df,
                pub_dir,
                loader
            )
            table_gen.generate_all(token_df)
            logger.info(f"   ✓ Tables saved to: {pub_dir}")
        else:
            logger.warning("\n⚠️  Step 5: Table generation skipped (dependencies missing)")
        
        # Step 6: Generate publication figures
        if FIGURES_AVAILABLE:
            logger.info("\n📈 Step 6: Generating publication figures...")
            fig_gen = FigureGenerator(
                perf_with_random,
                magic_with_random,
                perf_no_random,
                magic_no_random,
                features_df,
                pub_dir,
                loader
            )
            fig_gen.generate_all(token_df)
            logger.info(f"   ✓ Figures saved to: {pub_dir}")
        else:
            logger.warning("\n⚠️  Step 6: Figure generation skipped (dependencies missing)")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 ANALYSIS PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"📁 Output directory: {pub_dir}")
        logger.info("\nGenerated files:")
        
        if pub_dir.exists():
            csv_files = list(pub_dir.glob("*.csv"))
            png_files = list(pub_dir.glob("*.png"))
            logger.info(f"   📊 CSV files: {len(csv_files)}")
            logger.info(f"   📈 PNG files: {len(png_files)}")
            
            # List key files
            key_files = [
                "T_perf_win_rate.csv",
                "T_mlr_features_to_performance.csv",
                "T5_magic_to_perf.csv",
                "T7_combined_to_perf.csv",
                "T_similarity_3v5.csv"
            ]
            
            logger.info("\nKey output files:")
            for fname in key_files:
                fpath = pub_dir / fname
                if fpath.exists():
                    logger.info(f"   ✓ {fname}")
                else:
                    logger.info(f"   ✗ {fname} (not generated)")
        
        logger.info("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
