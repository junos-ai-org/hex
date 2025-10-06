# Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://junos-ai-org.github.io/hex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Jihoon Lee<sup>1</sup>, Hoyeon Moon<sup>1</sup>, Kevin Zhai<sup>5</sup>, Arun Kumar Chithanar, Anit Kumar Sahu<sup>2</sup>, Soummya Kar<sup>3</sup>, Chul Lee, Souradip Chakraborty<sup>4</sup>, Amrit Singh Bedi<sup>5</sup>**

<sup>1</sup>Yonsei University, <sup>2</sup>Oracle, <sup>3</sup>CMU, <sup>4</sup>UMD, <sup>5</sup>UCF

---
<p align="center">
  <img src="HEX/assets/HEX_visualization_sample.gif" alt="demo" />
</p>

<!-- Write anything below this line (left intentionally blank) -->
---
## Abstract

Diffusion-based large language models (dLLMs) are trained flexibly to model extreme dependence
in the data distribution; however, how to best utilize this information at inference time remains
an open problem. In this work, we uncover an interesting property of these models: dLLMs
trained on textual data implicitly learn a mixture of semi-autoregressive experts, where different
generation orders reveal different specialized behaviors. We show that committing to any single,
fixed inference time schedule, a common practice, collapses performance by failing to leverage
this latent ensemble. To address this, we introduce HEX (Hidden semiautoregressive EXperts for
test-time scaling), a training-free inference method that ensembles across heterogeneous block
schedules. By doing a majority vote over diverse block-sized generation paths, HEX robustly
avoids failure modes associated with any single fixed schedule. On reasoning benchmarks such
as GSM8K, it boosts accuracy by up to 3.56× (from 24.72% to 88.10%), outperforming top-K
margin inference and specialized fine-tuned methods like GRPO, without additional training. HEX
even yields significant gains on MATH benchmark from 16.40% to 40.00%, scientific reasoning on
ARC-C from 54.18% to 87.80%, and TruthfulQA from 28.36% to 57.46%. Our results establish a
new paradigm for test-time scaling in diffusion-based LLMs (dLLMs), revealing that the sequence
in which masking is performed plays a critical role in determining performance during inference.

## Key Features

- ✨ **Hidden Semi-Autoregressive Experts:** Reveals that diffusion LLMs implicitly learn multiple semi-AR experts, each specializing in distinct generation orders.
- 🚀 **Training-Free Test-Time Scaling:** Ensembles diverse block-sized decoding schedules at inference to unlock latent reasoning capabilities without retraining.

## Results

![Main Result](assets/figure_1.pdf)

## Installation

```bash
# Clone the repository
git clone https://github.com/junos-ai-org/Test-Time-Scaling
cd HEX

# Create a virtual environment
conda env create -f env.yml
conda activate dllm_tts
```

## Quick Start

Inside the `HEX/eval` directory, review the arguments described in `run_eval_HEX.sh` and run the script accordingly.

```bash
cd eval
bash run_eval_HEX.sh
```

## Project Structure

```
Test-Time-Scaling/
└── HEX/               # HEX Source code
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{lee2025hex,
  title={Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts},
  author={Lee, Jihoon and Moon, Hoyeon and Zhai, Kevin and Chithanar, Arun Kumar and Sahu, Anit Kumar and Kar, Soummya and Lee, Chul and Chakraborty, Souradip and Bedi, Amrit Singh},
  journal={Under Submission},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Most of the code of /HEX/ is based on [d1](https://github.com/dllm-reasoning/d1).

## Acknowledgments

[Optional: Add acknowledgments for funding, collaborators, or resources used]

## Contact

For questions or issues, please open an issue on GitHub or contact [email@domain.com]
