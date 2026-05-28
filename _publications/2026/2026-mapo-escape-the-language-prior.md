---
title:          "Escape the Language Prior: Mitigating Late-Stage Modality Collapse in Audio Reasoning via Modality-Aware Policy Optimization"
date:           2026-05-26 00:00:00 +0800
selected:       true
pub:            "arXiv preprint"
pub_date:       "2026"
cover: /assets/images/papers/MAPO.png
abstract: >-
  Audio and omni-modal large language models show strong cross-modal reasoning, but standard RL post-training like GRPO applies uniform policy gradients across all tokens, ignoring their unequal dependence on the non-text source modality. This worsens late-stage modality collapse during extended chain-of-thought, where models progressively abandon the primary source signal in favor of compressed textual priors, causing ungrounded hallucinations. We introduce MAPO, a novel dual-branch reinforcement learning framework. First, it concentrates the policy gradient on modality-critical tokens using a mask derived from cross-modal differential entropy between an audio-ablated reference and the multimodal policy. Second, it adds an auxiliary attention loss branch with a targeted, temporally scaled penalty to the model's internal attention distributions, ensuring sustained cross-modal grounding deep in the reasoning trace. Evaluations on complex audio reasoning benchmarks show MAPO improves long-horizon reasoning fidelity and instruction following, achieving new state-of-the-art results on several key benchmarks among open-weight models.
authors:
  - "Cihan Xiao"
  - "Yiwen Shao"
  - "Chenxing Li"
  - "Xiang He"
  - "Zhenwen Liang"
  - "Steve Yves"
  - "Sanjeev Khudanpur"
  - "Liefeng Bo"
links:
  arXiv: https://arxiv.org/abs/2605.27741
  PDF: https://arxiv.org/pdf/2605.27741.pdf
---
