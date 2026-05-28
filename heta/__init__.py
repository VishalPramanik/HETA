# Copyright (c) 2026 Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha
# University of Florida | University of Oklahoma | United States Military Academy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HETA: Hessian-Enhanced Token Attribution for Interpreting Autoregressive LLMs.

A principled attribution framework for decoder-only language models that integrates:
  (1) Semantic Transition Influence — causal attention-value flow tracing,
  (2) Hessian-Based Sensitivity    — second-order curvature via scalable HVPs,
  (3) KL-Based Information Impact   — distributional shift under token masking.

Reference:
    Pramanik, V., Maliha, M., Bastian, N. D., & Jha, S. K. (2026).
    Hessian-Enhanced Token Attribution (HETA): Interpreting Autoregressive LLMs.
    Published as a conference paper at ICLR 2026.
"""

__version__ = "1.0.0"
__author__ = "Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha"

from heta.attribution import HETA
from heta.semantic_flow import SemanticTransitionInfluence
from heta.hessian_sensitivity import HessianSensitivity
from heta.kl_divergence import KLInformationImpact
from heta.metrics import SoftNC, SoftNS, DSA
from heta.utils import HETAConfig, AttributionResult

__all__ = [
    "HETA",
    "HETAConfig",
    "AttributionResult",
    "SemanticTransitionInfluence",
    "HessianSensitivity",
    "KLInformationImpact",
    "SoftNC",
    "SoftNS",
    "DSA",
]
