"""Annotators for Lightricks' Union Control IC-LoRA: a reference video's frames as the depth and pose maps it reads,
made as Lightricks' LTX-2.5 Union Control workflow makes them (ComfyUI-LTXVideo
example_workflows/2.5/LTX-2.5_ICLoRA_Union_Control_Distilled.json, "Reference Video")."""

from .depth import VideoDepth
from .pose import PersonDetector, WholeBody, drawn
from .reference import scaled

__all__ = ["PersonDetector", "VideoDepth", "WholeBody", "drawn", "scaled"]
