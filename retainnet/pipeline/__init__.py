"""
Pipeline Module: 3-Stage Hybrid Architecture (Segmentation → Filtering → Survival Modeling).
"""
import os
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

from .stage1_segmentation import BehavioralSegmentationStage
from .stage2_filtering import InstanceFilteringStage
from .hybrid_pipeline import RetainNetHybridPipeline

__all__ = ["BehavioralSegmentationStage", "InstanceFilteringStage", "RetainNetHybridPipeline"]
