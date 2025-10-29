CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python src/train.py \
  --stage sft+ctl --model_family internvl3 \
  --base_model ../model/InternVL3-8B-hf \
  --sft_train_jsonl data/sft_example.jsonl \
  --sft_val_jsonl   data/sft_example.jsonl \
  --dpo_train_jsonl data/ctl_example.jsonl \
  --dpo_val_jsonl   data/ctl_example.jsonl \
  --image_prefix data/images
