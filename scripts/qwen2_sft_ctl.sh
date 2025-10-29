CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python src/train.py \
  --stage sft+ctl --model_family qwen2vl \
  --base_model ../model/Qwen2-VL-7B-Instruct \
  --sft_train_jsonl data/sft_example.jsonl \
  --sft_val_jsonl   data/sft_example.jsonl \
  --dpo_train_jsonl data/ctl_example.jsonl \
  --dpo_val_jsonl   data/ctl_example.jsonl \
  --image_prefix data/images
