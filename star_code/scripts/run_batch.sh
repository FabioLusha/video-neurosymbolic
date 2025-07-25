#!/bin/bash

python3 gemini_batch_creation.py \
	--input-dataset ../data/datasets/STAR/STAR_annotations/STAR_val_small_200.json \
	--limit-n 125 \
	--chunks 2 \
	--user-prompt ../data/prompts/vqa/user_prompt.txt \
	--videos-dir ../data/datasets/action-genome/Charades_v1_480 \
	--fps 1 \
	--max-frames 64 \
	--output-file ../scripts/chunked_gemini.jsonl
