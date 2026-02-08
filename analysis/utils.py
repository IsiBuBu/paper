"""
Utility functions for analysis pipeline.
"""

import pandas as pd
import numpy as np
from typing import List


def sig_stars(p_value: float) -> str:
    """Convert p-value to significance stars.
    
    Args:
        p_value: P-value from statistical test
        
    Returns:
        String with stars: '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), '' (ns)
    """
    if p_value is None or pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


def format_value(mean: float, std: float, sig: str = '') -> str:
    """Format mean ± std with optional significance stars.
    
    Args:
        mean: Mean value
        std: Standard deviation
        sig: Significance stars string
        
    Returns:
        Formatted string like "1.234 ± 0.056 **"
    """
    if pd.isna(mean):
        return "N/A"
    
    if pd.isna(std) or std == 0:
        base = f"{mean:.3f}"
    else:
        base = f"{mean:.3f} ± {std:.3f}"
    
    return f"{base} {sig}".strip()


def remove_collinear_predictors(df: pd.DataFrame, predictors: List[str], 
                                threshold: float = 0.95) -> List[str]:
    """Remove highly collinear predictors.
    
    When two predictors have correlation >= threshold, removes the one
    with lower variance (less informative).
    
    Args:
        df: DataFrame containing predictor columns
        predictors: List of predictor column names
        threshold: Correlation threshold for collinearity (default 0.95)
        
    Returns:
        List of predictors with collinear ones removed
    """
    # Filter to predictors with variance > 0
    remaining = [p for p in predictors if p in df.columns and df[p].std() > 0]
    
    while len(remaining) >= 2:
        corr = df[remaining].corr().abs()
        drop = None
        
        # Find first pair exceeding threshold
        for i in range(len(remaining)):
            for j in range(i+1, len(remaining)):
                if corr.iloc[i, j] >= threshold:
                    # Drop the one with lower variance
                    if df[remaining[i]].var() <= df[remaining[j]].var():
                        drop = remaining[i]
                    else:
                        drop = remaining[j]
                    break
            if drop:
                break
        
        if drop:
            remaining.remove(drop)
        else:
            break  # No more collinear pairs
    
    return remaining


# Game and metric configurations
GAME_CONFIGS = {
    'salop': {
        'performance_metrics': ['average_profit', 'win_rate', 'market_price'],
        'magic_metrics': ['rationality', 'reasoning', 'cooperation'],
        'game_specific_metric': 'market_price'
    },
    'spulber': {
        'performance_metrics': ['average_profit', 'win_rate', 'allocative_efficiency'],
        'magic_metrics': ['rationality', 'judgment', 'reasoning', 'self_awareness'],
        'game_specific_metric': 'allocative_efficiency'
    },
    'green_porter': {
        'performance_metrics': ['average_profit', 'win_rate', 'reversion_frequency'],
        'magic_metrics': ['cooperation', 'coordination'],
        'game_specific_metric': 'reversion_frequency'
    },
    'athey_bagwell': {
        'performance_metrics': ['average_profit', 'win_rate', 'productive_efficiency'],
        'magic_metrics': ['rationality', 'reasoning', 'deception', 'cooperation'],
        'game_specific_metric': 'productive_efficiency'
    }
}

METRIC_DIRECTION = {
    'average_profit': '↑',
    'win_rate': '↑',
    'market_price': '↑',
    'allocative_efficiency': '↑',
    'productive_efficiency': '↑',
    'reversion_frequency': '↓',
    'rationality': '↑',
    'reasoning': '↑',
    'cooperation': '↑',
    'coordination': '↑',
    'judgment': '↑',
    'self_awareness': '↑',
    'deception': '↑',
}

MODEL_FEATURES = [
    'architecture_moe',
    'size_params',
    'family_encoded',
    'family_version',
    'thinking'
]

COLLINEARITY_THRESHOLD = 0.95
