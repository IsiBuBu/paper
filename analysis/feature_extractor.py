"""
Model feature extraction module.
"""

import pandas as pd
import re
from typing import List, Dict
import logging

try:
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts architectural and configuration features from model names."""
    
    def __init__(self, model_configs: Dict):
        self.model_configs = model_configs
        self.family_encoder = None
    
    def extract_features(self, models: List[str]) -> pd.DataFrame:
        """Extract features from model names.
        
        Features extracted:
        - architecture_moe: Binary indicator for Mixture-of-Experts models
        - size_params: Model size in billions of parameters
        - family: Model family (qwen, llama, etc.)
        - family_version: Within-family version (0, 1, 2 for Llama-3.1, 3.3, 4)
        - thinking: Binary indicator for reasoning/thinking mode
        - family_encoded: Integer-encoded family for regression
        
        Args:
            models: List of model names/IDs
            
        Returns:
            DataFrame with one row per model and feature columns
        """
        records = []
        
        for model in models:
            m = str(model).lower()
            
            # Detect Mixture-of-Experts architecture
            is_moe = bool(re.search(r'-a\d+b|moe|maverick|scout', m))
            
            # Extract parameter size
            size = self._extract_size(m, is_moe)
            
            # Extract family
            family = self._extract_family(m)
            
            # Extract within-family version
            family_version = self._extract_family_version(m)
            
            # Get thinking mode from config
            thinking = self._get_thinking_mode(model)
            
            records.append({
                'model': model,
                'architecture_moe': int(is_moe),
                'size_params': size,
                'family': family,
                'family_version': family_version,
                'thinking': thinking
            })
        
        df = pd.DataFrame(records)
        
        # Encode family as integer for regression
        if SKLEARN_AVAILABLE and len(df) > 0:
            self.family_encoder = LabelEncoder()
            df['family_encoded'] = self.family_encoder.fit_transform(df['family'])
        else:
            # Fallback: manual encoding
            families = df['family'].unique()
            family_map = {f: i for i, f in enumerate(sorted(families))}
            df['family_encoded'] = df['family'].map(family_map)
        
        logger.info(f"Extracted features for {len(df)} models")
        return df
    
    def _extract_size(self, model_lower: str, is_moe: bool) -> float:
        """Extract model size in billions of parameters."""
        if is_moe:
            # Qwen3 MoE: Extract active parameters from -A\d+B pattern
            qwen_active = re.search(r'-a(\d+\.?\d*)b', model_lower)
            if qwen_active:
                return float(qwen_active.group(1))
            
            # Llama MoE: Extract main parameter, ignore expert count
            llama_size = re.search(r'(?<!a)(\d+\.?\d*)b(?!.*-\d+e)', model_lower)
            if not llama_size:
                # Fallback: find any \d+B not preceded by 'a'
                llama_size = re.search(r'(?<!a)(\d+\.?\d*)b', model_lower)
            return float(llama_size.group(1)) if llama_size else 0.0
        else:
            # Standard models: use first \d+B not preceded by 'a'
            standard_size = re.search(r'(?<!a)(\d+\.?\d*)b', model_lower)
            return float(standard_size.group(1)) if standard_size else 0.0
    
    def _extract_family(self, model_lower: str) -> str:
        """Extract model family."""
        families = ['qwen', 'llama', 'gemma', 'mistral', 'gemini', 'gpt', 'claude']
        for family in families:
            if family in model_lower:
                return family
        return 'unknown'
    
    def _extract_family_version(self, model_lower: str) -> int:
        """Extract within-family version (ordinal encoding).
        
        For Llama models:
        - Llama-3.1: 0 (oldest)
        - Llama-3.3: 1 (middle)
        - Llama-4: 2 (newest)
        
        For other families: 0 (single generation)
        """
        if 'llama' in model_lower:
            if re.search(r'llama-?3\.1', model_lower):
                return 0
            elif re.search(r'llama-?3\.3', model_lower):
                return 1
            elif re.search(r'llama-?4', model_lower):
                return 2
        return 0
    
    def _get_thinking_mode(self, model: str) -> int:
        """Get thinking mode from config (1 if enabled, 0 otherwise)."""
        reasoning_output = self.model_configs.get(model, {}).get('reasoning_output', 'none')
        return 1 if reasoning_output != 'none' else 0
