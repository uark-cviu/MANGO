# [NeurIPS 2025] MANGO: Multimodal Attention-based Normalizing Flow Approach to Fusion Learning

[![Paper](https://img.shields.io/badge/arXiv-2305.15700-brightgreen)](https://arxiv.org/abs/2508.10133)
[![Conference](https://img.shields.io/badge/NeurIPS-2025-blue)](https://arxiv.org/abs/2508.10133)

> [MANGO: Multimodal Attention-based Normalizing Flow Approach to Fusion Learning](https://arxiv.org/abs/2508.10133)<br>
> [Thanh-Dat Truong](https://truongthanhdat.github.io/), [Christophe Bobda](), [Nitin Agarwal](), and [Khoa Luu](http://csce.uark.edu/~khoaluu)<br>
> University of Arkansas, Computer Vision and Image Understanding Lab, CVIU<br>

## Abstract

Multimodal learning has gained much success in recent years. However, current multimodal fusion methods adopt the attention mechanism of Transformers to implicitly learn the underlying correlation of multimodal features. As a result, the multimodal model cannot capture the essential features of each modality, making it difficult to comprehend complex structures and correlations of multimodal inputs. This paper introduces a novel Multimodal Attention-based Normalizing Flow (MANGO) approach to developing explicit, interpretable, and tractable multimodal fusion learning. In particular, we propose a new Invertible Cross-Attention (ICA) layer to develop the Normalizing Flow-based Model for multimodal data. To efficiently capture the complex, underlying correlations in multimodal data in our proposed invertible cross-attention layer, we propose three new cross-attention mechanisms: Modality-to-Modality Cross-Attention (MMCA), Inter-Modality Cross-Attention (IMCA), and Learnable Inter-Modality Cross-Attention (LICA). Finally, we introduce a new Multimodal Attention-based Normalizing Flow to enable the scalability of our proposed method to high-dimensional multimodal data. Our experimental results on three different multimodal learning tasks, i.e., semantic segmentation, image-to-image translation, and movie genre classification, have illustrated the state-of-the-art (SoTA) performance of the proposed approach.



## Training and Testing

The training and testing code will be released soon.

## Acknowledgements

A part of the codebase of this project is borrowed from [TokenFusion](https://github.com/yikaiw/TokenFusion).

This work is partly supported by NSF CAREER (No. 2442295), NSF SCH (No. 2501021), NSF E-RISE (No. 2445877), NSF SBIR Phase 2 (No. 2247237) and USDA/NIFA Award. We also acknowledge the Arkansas High-Performance Computing Center (HPC) for GPU servers.

## Citation

If you find this code useful for your research, please consider citing:
```
@article{truong2025mango,
  title={MANGO: Multimodal Attention-based Normalizing Flow Approach to Fusion Learning},
  author={Thanh-Dat Truong and Christophe Bobda and Nitin Agarwal and Khoa Luu},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

