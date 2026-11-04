"""
RetainNet: Deep Survival Analysis Framework for Dynamic CLTV & Churn Mitigation.
"""
__version__ = "1.0.0"

import os
import warnings
import importlib
from typing import Any

# Prevent Windows loky subprocess physical core detection warning
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")

_LAZY_EXPORTS = {
    "SyntheticSurvivalDataGenerator": "retainnet.data",
    "DataPreprocessingPipeline": "retainnet.data",
    "TelcoDatasetLoader": "retainnet.data",
    "validate_customer_csv": "retainnet.data",
    "generate_csv_template": "retainnet.data",
    "ValidationResult": "retainnet.data",
    "RetainNetHybridPipeline": "retainnet.pipeline",
    "BehavioralSegmentationStage": "retainnet.pipeline",
    "InstanceFilteringStage": "retainnet.pipeline",
    "CoxPHBaselineModel": "retainnet.models",
    "RandomSurvivalForestBaselineModel": "retainnet.models",
    "DeepSurvModel": "retainnet.models",
    "DeepHitModel": "retainnet.models",
    "Evaluator": "retainnet.metrics",
    "DynamicLTVCalculator": "retainnet.ltv",
    "DriftDetector": "retainnet.advanced",
    "ConformalSurvivalCalibrator": "retainnet.advanced",
    "AmortizedSurvivalInference": "retainnet.advanced",
}

__all__ = [
    "__version__",
    "SyntheticSurvivalDataGenerator",
    "DataPreprocessingPipeline",
    "TelcoDatasetLoader",
    "validate_customer_csv",
    "generate_csv_template",
    "ValidationResult",
    "RetainNetHybridPipeline",
    "BehavioralSegmentationStage",
    "InstanceFilteringStage",
    "CoxPHBaselineModel",
    "RandomSurvivalForestBaselineModel",
    "DeepSurvModel",
    "DeepHitModel",
    "Evaluator",
    "DynamicLTVCalculator",
    "DriftDetector",
    "ConformalSurvivalCalibrator",
    "AmortizedSurvivalInference",
]

def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod = importlib.import_module(_LAZY_EXPORTS[name])
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return __all__
