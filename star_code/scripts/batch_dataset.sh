#!/bin/bash

python3 gemini_batch_creation.py \
	--input-dataset ../data/datasets/STAR/STAR_annotations/STAR_val_small_200.json \
	--limit-n 200 \
	--chunks 2 \
	--task "vqa" \
	--user-prompt ../data/prompts/vqa/user_prompt.txt \
	--videos-dir ../data/datasets/action-genome/Charades_v1_480 \
	--fps 1 \
	--max-frames 64 \
	--output-file tmp/chunked_gemini.jsonl
