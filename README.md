# BEAT: Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning


</p>
<p align="left">
  <a href='https://arxiv.org/abs/2510.27623'>
    <img src='https://img.shields.io/badge/Arxiv-2510.27623-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://zqs1943.github.io/BEAT/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
</p>

## 🏠 Overview
https://github.com/user-attachments/assets/b096e582-6e8c-4d5c-8335-2efd78df99b7

**BEAT** is the first to show visual backdoors in MLLM embodied agents: fine-tune the MLLM to implant a backdoor so the agent behaves normally until a specific object trigger, then follows an attacker-specified policy.

<div align="center">
  <img src="assets/method.png" width="100%" alt="InterMimic teaser"/>
</div>

**BEAT** uses a two-stage training pipeline: (i) standard supervised fine-tuning (SFT) on a mixture of benign and backdoor trajectories to strengthen general capabilities, followed by (ii) our Contrastive Trigger Learning (CTL), a preference-learning procedure that improves the precision of backdoor activation.


## ⚒️ Environment Setup
```bash
conda create -n beat python=3.10
conda activate beat
pip install -r requirements.txt
```

## 🗂️ Data Preparation
We provide example fine-tuning data for SFT and CTL in `./data`. Each SFT example consists of an input (history plus image) and the MLLM’s target output. Each CTL example is a contrastive pair identical except for trigger presence in the image and the associated target output. Due to ethical considerations, the full training set is available upon request.

## 🎛️ BEAT Finetuning (SFT + CTL)
We prepare the scripts of running BEAT finetuning over the on the example dataset:
```bash
bash scripts/qwen2_sft_ctl.sh
bash scripts/internvl_sft_ctl.sh
```
To run on other model, you need to customized the llm finetuning interface in `src/llms`.

## Citation
```bibtex
@article{zhan2025beat,
    title={Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning},
    author={Zhan, Qiusi and Ha, Hyeonjeong and Yang, Rui and Xu, Sirui and Chen, Hanyang and Gui, Liang-yan and Wang, Yu-Xiong and Zhang, Huan and Ji, Heng and Kang, Daniel}, 
    year={2025}
}
```