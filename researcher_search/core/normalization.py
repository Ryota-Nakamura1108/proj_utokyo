"""Robust normalization utilities for handling heavy-tailed distributions and outliers."""

import math
import logging
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Available normalization methods."""
    STANDARD = "standard"  # Mean/std normalization
    ROBUST_Z = "robust_z"  # Median/MAD normalization  
    RANKIT = "rankit"      # Percentile rank to normal quantile transformation


class RobustNormalizer:
    """Unified robust normalization utilities for heavy-tailed distributions."""
    
    def __init__(self, method: NormalizationMethod = NormalizationMethod.ROBUST_Z):
        """Initialize normalizer with specified method.
        
        Args:
            method: Normalization method to use
        """
        self.method = method
    
    def normalize_values(self, values: List[float], 
                        method: Optional[NormalizationMethod] = None) -> List[float]:
        """Normalize a list of values using the specified method.
        
        Args:
            values: List of values to normalize
            method: Optional override for normalization method
            
        Returns:
            List of normalized values in range [0, 1]
        """
        if not values:
            return []
        
        # Filter out NaN/None values
        clean_values = [v for v in values if v is not None and not math.isnan(v) if isinstance(v, (int, float))]
        if not clean_values:
            return [0.5] * len(values)  # All invalid values -> neutral
        
        normalization_method = method or self.method
        
        if normalization_method == NormalizationMethod.STANDARD:
            return self._standard_normalize(clean_values)
        elif normalization_method == NormalizationMethod.ROBUST_Z:
            return self._robust_z_normalize(clean_values)
        elif normalization_method == NormalizationMethod.RANKIT:
            return self._rankit_normalize(clean_values)
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")
    
    def normalize_dict_values(self, data_dict: Dict[str, float],
                            method: Optional[NormalizationMethod] = None) -> Dict[str, float]:
        """Normalize values in a dictionary.
        
        Args:
            data_dict: Dictionary mapping keys to values
            method: Optional override for normalization method
            
        Returns:
            Dictionary with normalized values
        """
        if not data_dict:
            return {}
        
        values = list(data_dict.values())
        normalized_values = self.normalize_values(values, method)
        
        return {key: norm_val for key, norm_val in zip(data_dict.keys(), normalized_values)}
    
    def _standard_normalize(self, values: List[float]) -> List[float]:
        """Standard mean/std normalization with sigmoid transformation."""
        if len(values) <= 1:
            return [0.5] * len(values)
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return [0.5] * len(values)
        
        # Z-score then sigmoid transform to [0, 1]
        normalized = []
        for value in values:
            z_score = (value - mean_val) / std_val
            norm_value = 1.0 / (1.0 + math.exp(-z_score))  # Sigmoid
            normalized.append(norm_value)
        
        return normalized
    
    def _robust_z_normalize(self, values: List[float]) -> List[float]:
        """Robust normalization using median and MAD (Median Absolute Deviation)."""
        if len(values) <= 1:
            return [0.5] * len(values)
        
        median = np.median(values)
        mad = np.median(np.abs(np.array(values) - median))
        
        if mad == 0:
            return [0.5] * len(values)  # All values are the same
        
        # Robust z-score with scaling factor for normal distribution equivalence
        normalized = []
        for value in values:
            robust_z = (value - median) / (1.4826 * mad)
            norm_value = 1.0 / (1.0 + math.exp(-robust_z))  # Sigmoid
            normalized.append(norm_value)
        
        return normalized
    
    def _rankit_normalize(self, values: List[float]) -> List[float]:
        """Rankit (percentile rank to normal quantile) transformation.
        
        This method is particularly effective for heavy-tailed distributions
        as it equalizes the influence of extreme values.
        """
        if len(values) <= 1:
            return [0.5] * len(values)
        
        # Calculate percentile ranks
        sorted_indices = np.argsort(values)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(len(values))
        
        # Convert ranks to percentiles (0 to 1)
        percentiles = (ranks + 0.5) / len(values)  # +0.5 to avoid 0 and 1 exactly
        
        # Transform percentiles to normal quantiles, then to [0, 1] via CDF
        normalized = []
        for p in percentiles:
            # Clamp percentiles to avoid extreme values
            p_clamped = max(0.001, min(0.999, p))
            
            # Normal quantile (inverse CDF)
            z_score = stats.norm.ppf(p_clamped)
            
            # Transform back to [0, 1] using CDF with reasonable bounds
            norm_value = stats.norm.cdf(z_score / 2)  # Scale down for better distribution
            normalized.append(norm_value)
        
        return normalized
    
    def calculate_normalization_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics for normalization assessment.
        
        Args:
            values: List of values to analyze
            
        Returns:
            Dictionary with statistics useful for choosing normalization method
        """
        if not values:
            return {}
        
        clean_values = np.array([v for v in values if v is not None and not math.isnan(v)])
        if len(clean_values) == 0:
            return {}
        
        stats_dict = {
            'count': len(clean_values),
            'mean': float(np.mean(clean_values)),
            'median': float(np.median(clean_values)),
            'std': float(np.std(clean_values)),
            'mad': float(np.median(np.abs(clean_values - np.median(clean_values)))),
            'min': float(np.min(clean_values)),
            'max': float(np.max(clean_values)),
            'q25': float(np.percentile(clean_values, 25)),
            'q75': float(np.percentile(clean_values, 75)),
            'skewness': float(stats.skew(clean_values)),
            'kurtosis': float(stats.kurtosis(clean_values))
        }
        
        # Calculate robust-to-standard ratio (indicator of heavy tails/outliers)
        if stats_dict['std'] > 0 and stats_dict['mad'] > 0:
            stats_dict['robust_ratio'] = (stats_dict['mad'] * 1.4826) / stats_dict['std']
        else:
            stats_dict['robust_ratio'] = 1.0
        
        return stats_dict
    
    @staticmethod
    def recommend_normalization_method(values: List[float]) -> NormalizationMethod:
        """Recommend the best normalization method based on data characteristics.
        
        Args:
            values: List of values to analyze
            
        Returns:
            Recommended normalization method
        """
        normalizer = RobustNormalizer()
        stats_dict = normalizer.calculate_normalization_stats(values)
        
        if not stats_dict:
            return NormalizationMethod.ROBUST_Z  # Default fallback
        
        # Decision criteria based on distribution characteristics
        skewness = abs(stats_dict.get('skewness', 0))
        kurtosis = stats_dict.get('kurtosis', 0)
        robust_ratio = stats_dict.get('robust_ratio', 1.0)
        
        # Heavy tails or extreme outliers → rankit
        if kurtosis > 5 or robust_ratio < 0.5:
            return NormalizationMethod.RANKIT
        
        # Moderate skewness or outliers → robust z-score
        elif skewness > 1.5 or robust_ratio < 0.8:
            return NormalizationMethod.ROBUST_Z
        
        # Approximately normal → standard normalization
        else:
            return NormalizationMethod.STANDARD
    
    def normalize_metrics_dict(self, 
                              metrics: Dict[str, Dict[str, float]],
                              method_overrides: Optional[Dict[str, NormalizationMethod]] = None) -> Dict[str, Dict[str, float]]:
        """Normalize metrics across multiple authors with method selection per metric.
        
        Args:
            metrics: Dict of {author_id: {metric_name: value}}
            method_overrides: Optional dict of {metric_name: NormalizationMethod}
            
        Returns:
            Normalized metrics with same structure
        """
        if not metrics:
            return {}
        
        # Collect values by metric
        metric_values = {}
        for author_metrics in metrics.values():
            for metric_name, value in author_metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                if isinstance(value, (int, float)) and not math.isnan(value):
                    metric_values[metric_name].append(value)
        
        # Normalize each metric
        normalized_metrics = {}
        for author_id, author_data in metrics.items():
            normalized_metrics[author_id] = {}
            
            for metric_name, value in author_data.items():
                if metric_name in metric_values and len(metric_values[metric_name]) > 0:
                    # Determine normalization method
                    if method_overrides and metric_name in method_overrides:
                        method = method_overrides[metric_name]
                    else:
                        method = self.recommend_normalization_method(metric_values[metric_name])
                    
                    # Normalize single value within context of all values for this metric
                    all_values = metric_values[metric_name]
                    normalized_all = self.normalize_values(all_values, method)
                    
                    # Find the normalized value for this specific author
                    try:
                        value_index = all_values.index(value)
                        normalized_metrics[author_id][metric_name] = normalized_all[value_index]
                    except ValueError:
                        # Value not found (e.g., NaN), use neutral value
                        normalized_metrics[author_id][metric_name] = 0.5
                else:
                    # No valid values for this metric
                    normalized_metrics[author_id][metric_name] = 0.5
        
        return normalized_metrics