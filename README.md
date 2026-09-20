# BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning


</p>
<p align="left">
  <a href='https://arxiv.org/abs/2510.27623'>
    <img src='https://img.shields.io/badge/Arxiv-2510.27623-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://zqs1943.github.io/BEAT/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://huggingface.co/datasets/uiuc-kang-lab/BEAT'>
    <img src='https://img.shields.io/badge/Dataset-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=FFD21E'></a>
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
We provide example fine-tuning data for SFT and CTL in `./data`. Each SFT example consists of an input (history plus image) and the VLM’s target output. Each CTL example is a contrastive pair identical except for trigger presence in the image and the associated target output. The full training dataset is available at [HuggingFace](https://huggingface.co/datasets/uiuc-kang-lab/BEAT).

## 🎛️ BEAT Finetuning (SFT + CTL)
We prepare the scripts of running BEAT finetuning over the on the example dataset:
```bash
bash scripts/qwen2_sft_ctl.sh
bash scripts/internvl_sft_ctl.sh
```
To run on other model, you need to customized the llm finetuning interface in `src/llms`.

## 🤖 Embodied Environment (VAB-OmniGibson)

`./vab_omnigibson` contains everything needed to reproduce the environment side of BEAT: planting the
visual trigger into [VAB-OmniGibson](https://github.com/THUDM/VisualAgentBench), rolling out agents in
the poisoned scenes, and scoring backdoor activation. This is where the trajectories that feed the SFT
and CTL stages above come from.

BEAT is a small set of edits on top of an unmodified VAB checkout, not a fork, so it ships as both a
unified diff and the equivalent whole files:

| | |
|---|---|
| `beat.patch` | Unified diff vs. upstream VAB — 36 files |
| `patch/` | The same edits as whole files, mirroring VAB's tree |
| `trigger/` | The visual trigger: a pink carving knife (USD asset + recoloring script) |
| `scene_generation/` | 21 backdoor + 5 OOD reference scenes, and the scripts that expand them |
| `evaluation/` | Rollout collection, ASR / benign-SR scoring, and CTL pair construction |

```bash
git clone https://github.com/THUDM/VisualAgentBench.git && cd VisualAgentBench
git apply /path/to/BEAT/vab_omnigibson/beat.patch
```

Note that `vab_omnigibson/` ships the *inputs* to the poisoned dataset, not the dataset itself. The
trigger asset, the 21 backdoor reference scenes and the task registry are included, but the ~100
expanded backdoor scenes and their task definitions are derived data that you generate locally,
against your own OmniGibson dataset download. Agent rollout trajectories are likewise not included —
the released training data lives on [HuggingFace](https://huggingface.co/datasets/uiuc-kang-lab/BEAT).

See [`vab_omnigibson/README.md`](vab_omnigibson/README.md) for the full setup, how the trigger and the
scripted attacker policy work, what you need to rebuild, and known issues.

## Citation
```bibtex
@inproceedings{zhan2026beat,
  title = {BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning},
  author = {Zhan, Qiusi and Ha, Hyeonjeong and Yang, Rui and Xu, Sirui and Chen, Hanyang and Gui, Liang-Yan and Wang, Yu-Xiong and Zhang, Huan and Ji, Heng and Kang, Daniel},
  booktitle = {ICLR},
  year = {2026},
}
```