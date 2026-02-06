# BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning


</p>
<p align="left">
  <a href='https://arxiv.org/abs/2510.27623'>
    <img src='https://img.shields.io/badge/Arxiv-2510.27623-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://zqs1943.github.io/BEAT/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
</p>

## 🏠 Overview
https://github.com/user-attachments/assets/26b9c564-e34a-422f-86e8-4065453c0916

**BEAT** is the first to show visual backdoors in VLM-based embodied agents: fine-tune the VLM to implant a backdoor so the agent behaves normally until a specific object trigger, then follows an attacker-specified policy.

https://github.com/user-attachments/assets/329086a8-bdc9-4834-a496-715bdf3719a5

**BEAT** uses a two-stage training pipeline: (i) standard supervised fine-tuning (SFT) on a mixture of benign and backdoor trajectories to strengthen general capabilities, followed by (ii) our Contrastive Trigger Learning (CTL), a preference-learning procedure that improves the precision of backdoor activation.


## ⚒️ Environment Setup
```bash
conda create -n beat python=3.10
conda activate beat
pip install -r requirements.txt
```

## 🗂️ Data Preparation
We provide example fine-tuning data for SFT and CTL in `./data`. Each SFT example consists of an input (history plus image) and the VLM’s target output. Each CTL example is a contrastive pair identical except for trigger presence in the image and the associated target output. Due to ethical considerations, the full training set is available upon request.

## 🎛️ BEAT Finetuning (SFT + CTL)
We prepare the scripts of running BEAT finetuning over the on the example dataset:
```bash
bash scripts/qwen2_sft_ctl.sh
bash scripts/internvl_sft_ctl.sh
```
To run on other model, you need to customized the llm finetuning interface in `src/llms`.

## Citation
```bibtex
@inproceedings{zhan2026beat,
  title = {BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning},
  author = {Zhan, Qiusi and Ha, Hyeonjeong and Yang, Rui and Xu, Sirui and Chen, Hanyang and Gui, Liang-Yan and Wang, Yu-Xiong and Zhang, Huan and Ji, Heng and Kang, Daniel},
  booktitle = {ICLR},
  year = {2026},
}
```