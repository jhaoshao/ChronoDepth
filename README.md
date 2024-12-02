# ChronoDepth: Learning Temporally Consistent Video Depth from Video Diffusion Priors

This repository represents the official implementation of the paper titled "Learning Temporally Consistent Video Depth from Video Diffusion Priors".

[![Website](https://img.shields.io/website?url=https%3A%2F%2Fjhaoshao.github.io%2FChronoDepth%2F&up_message=ChronoDepth&up_color=blue&style=flat&logo=timescale&logoColor=%23FFDC0F)](https://xdimlab.github.io/ChronoDepth/) [![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2406.01493) [![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/jhshao/ChronoDepth)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/jhshao/ChronoDepth-v1)

[Jiahao Shao*](https://jhaoshao.github.io/), Yuanbo Yang*, Hongyu Zhou, [Youmin Zhang](https://youmi-zym.github.io/),  [Yujun Shen](https://shenyujun.github.io/), [Vitor Guizilini](https://vitorguizilini.github.io/), [Yue Wang](https://yuewang.xyz/), [Matteo Poggi](https://mattpoggi.github.io/), [Yiyi Liao](https://yiyiliao.github.io/ )

## 📢 News
2024-12-03: Release inference code and checkpoint for new version <a href="https://huggingface.co/jhshao/ChronoDepth-v1"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green" height="16"></a>
2024-06-11: Added <a href="https://huggingface.co/spaces/jhshao/ChronoDepth"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow" height="16"></a> - try it out with your videos for free!<br>2024-06-11: Added <a href="https://arxiv.org/abs/2406.01493"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" height="16"></a> paper and inference code (this repository).


## 🛠️ Setup
We test our codes under the following environment: `Ubuntu 22.04, Python 3.10.15, CUDA 12.1, RTX A6000`.
1. Clone this repository.
```bash
git clone https://github.com/jhaoshao/ChronoDepth
cd ChronoDepth
```
2. Install packages
```bash
conda create -n chronodepth python=3.10 -y
conda activate chronodepth
pip install -r requirements.txt
```

## 🚀 Run inference
Run the python script `run_infer.py` as follows
```bash
python run_infer.py \
    --unet=jhshao/ChronoDepth-v1 \
    --model_base=stabilityai/stable-video-diffusion-img2vid-xt \
    --seed=1234 \
    --data_dir=assets/elephant.mp4 \
    --output_dir=./outputs \
    --denoise_steps=5 \
    --chunk_size=5 \
    --n_tokens=10
```
Inference settings:
- `--denoise_steps`: the number of steps for the denoising process.
- `--chunk_size`: chunk size of sliding window for sliding window inference.
- `--n_tokens`: number of frames of each clip for sliding window inference.

## ✅ TODO
- [x] Release inference code and checkpoint for new version
- [ ] Set up Online demo for new version
- [ ] Release evaluation code
- [ ] Release training code & dataset preparation

## 🎓 Citation

Please cite our paper if you find this repository useful:

```bibtex
@misc{shao2024learning,
      title={Learning Temporally Consistent Video Depth from Video Diffusion Priors}, 
      author={Jiahao Shao and Yuanbo Yang and Hongyu Zhou and Youmin Zhang and Yujun Shen and Matteo Poggi and Yiyi Liao},
      year={2024},
      eprint={2406.01493},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```