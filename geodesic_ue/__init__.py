"""Utility Engineering analyses applied to Geodesic's alignment-pretraining suite.

The vendored emergent-values checkout (repo root /emergent-values) is treated as
READ-ONLY. We import its pure-torch Thurstonian fitting code by file path (its
package import chain drags in vllm/litellm/anthropic/etc., which we neither need
nor want on every machine); the handful of small pure-Python helpers whose home
modules have those heavy import chains are copied verbatim into vendor.py with
attribution. Every deviation from UE defaults is recorded in NOTES.md.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UE_ROOT = os.path.join(REPO_ROOT, "emergent-values", "utility_analysis")
UE_OPTIONS_PATH = os.path.join(UE_ROOT, "shared_options", "options_hierarchical.json")
