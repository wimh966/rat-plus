# RAT+

Official implementation of **RAT+**, a dense-pretraining architecture that augments attention with full-sequence recurrence and active recurrence learning.

A single RAT+ model is pretrained densely once and can then be flexibly switched at inference time to dilated attention (optionally with local windows) or hybrid layer/head compositions. This requires only a short 1B-token resolution adaptation rather than retraining separate sparse models.

This repository currently provides the core architecture implementation. The full codebase, including training scripts and evaluation pipelines, is currently being cleaned up and will be released in a future update of this repository.

<p align="center">
  <img src="assets/method.png" width="750">
</p>

# Citation

If you find this work useful, please cite:

The repository structure is built upon https://github.com/CLAIRE-Labo/RAT.

```bibtex
@article{wei2025rat,
  title={RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling},
  author={Wei, Xiuying and Yadav, Anunay and Pascanu, Razvan and Gulcehre, Caglar},
  journal={arXiv preprint arXiv:2507.04416},
  year={2025}
}

@article{wei2026ratplus,
  title={RAT+: Train Dense, Infer Sparse--Recurrence Augmented Attention for Dilated Inference},
  author={Wei, Xiuying and Gulcehre, Caglar},
  journal={arXiv preprint arXiv:2602.18196},
  year={2026}
}