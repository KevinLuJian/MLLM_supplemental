python inference_code/DVQA_evaluations.py --model MGM-7B --device cuda:0 --answer evaluation_results/DVQA/dvqa_MGM-7B.jsonl
python inference_code/tallyqa_evaluations.py --model MGM-7B --device cuda:1 --answer evaluation_results/TallyQA/TallyQA_MGM-7B.jsonl
python inference_code/TDIUC_evaluations.py --model MGM-7B --device cuda:2 --answer evaluation_results/TDIUC/TDIUC_MGM-7B.jsonl
python inference_code/VQDv1_evaluation.py --model MGM-7B --device cuda:3 --answer evaluation_results/VQDv1/VQDv1_MGM-7B.jsonl
