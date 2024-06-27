# ChronoDepth: Learning Temporally Consistent Video Depth from Video Diffusion Priors

This repository represents the official implementation of the paper titled "Learning Temporally Consistent Video Depth from Video Diffusion Priors".

[![Website](https://img.shields.io/website?url=https%3A%2F%2Fjhaoshao.github.io%2FChronoDepth%2F&up_message=ChronoDepth&up_color=blue&style=flat&logo=timescale&logoColor=%23FFDC0F)](https://jhaoshao.github.io/ChronoDepth/) [![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2406.01493) [![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/jhshao/ChronoDepth)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/jhshao/ChronoDepth)

[Jiahao Shao*](https://jhaoshao.github.io/), Yuanbo Yang*, Hongyu Zhou, [Youmin Zhang](https://youmi-zym.github.io/),  [Yujun Shen](https://shenyujun.github.io/), [Matteo Poggi](https://mattpoggi.github.io/), [Yiyi Liao†](https://yiyiliao.github.io/ )

## 📢 News
2024-06-11: Added <a href="https://huggingface.co/spaces/jhshao/ChronoDepth"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow" height="16"></a> - try it out with your videos for free!<br>2024-06-11: Added <a href="https://arxiv.org/abs/2406.01493"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" height="16"></a> paper and inference code (this repository).

## 🛠️ Setup
We test our codes under the following environment: `Ubuntu 20.04, Python 3.10.14, CUDA 11.3, RTX A6000`.
1. Clone this repository.
```bash
git clone https://github.com/jhaoshao/ChronoDepth
cd ChronoDepth
```
2. Install packages
```bash
conda create -n chronodepth python=3.10
conda activate chronodepth
pip install -r requirements.txt
```

## 🚀 Run inference
Run the python script `run_infer.py` as follows
```bash
python run_infer.py \
    --model_base=checkpoints/ChronoDepth \
    --data_dir=assets/sora_e2.mp4 \
    --output_dir=./outputs \
    --num_frames=10 \
    --denoise_steps=10 \
    --window_size=9 \
    --half_precision \
    --seed=1234 \
```
Inference settings:
- `--num_frames`: sets the number of frames for each video clip.
- `--denoise_steps`: sets the number of steps for the denoising process.
- `--window_size`: sets the size of sliding window. This implies conducting separate inference when the sliding window size equals the number of frames.
- `--half_precision`: enables running with half-precision (16-bit float). It might lead to suboptimal result but could speed up the inference process.

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