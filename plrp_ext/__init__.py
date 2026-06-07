"""
PLRP extension for the CleverHansRegression PRP-INSightR-Net pipeline.

This package adds Pruned Layer-wise Relevance Propagation (PLRP-lambda;
Yanez Sarmiento et al., ECML PKDD 2024) on top of the existing PRP
implementation, *without* modifying any existing file.

Standard PRP can still be run via the parent-folder scripts
(main_generate_prp.py, insight_prp.py, lrp_general6.py) exactly as before.

PLRP can be enabled by calling `set_plrp_params` before canonization:
    plrp_ext.lrp_general6_plrp.set_plrp_params(p_pos=0.25, p_neg=0.125)

When p_pos == p_neg == 0 (the default), this package's behaviour is
mathematically identical to the standard PRP pipeline. This invariant is
verified by tests/test_plrp_equivalence.py.
"""
