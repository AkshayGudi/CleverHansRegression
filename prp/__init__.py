"""
Baseline Prototypical Relevance Propagation (PRP) for INSightR-Net.

Core modules:
  - insight_prp: PRP heatmap generation and canonized model
  - lrp_general6: LRP propagation rules used by PRP
  - main_generate_prp: CLI entry point for PRP heatmaps
  - relevance_ordering_*: insertion-based faithfulness evaluation
  - localization_metrics: GT-mask localization metrics (PG, RMA, RRA, AUC, IoU)
"""
