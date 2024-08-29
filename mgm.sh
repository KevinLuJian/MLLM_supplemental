#export CUDA_HOME=/scratch/ssrivas9/miniconda3/envs/mgm

# MGM 7B
python inference_code/DVQA_evaluations.py --model MGM-7B --device cuda:0 --answer evaluation_results/DVQA/dvqa_MGM-7B.jsonl
python inference_code/tallyqa_evaluations.py --model MGM-7B --device cuda:1 --answer evaluation_results/TallyQA/TallyQA_MGM-7B.jsonl
python inference_code/TDIUC_evaluations.py --model MGM-7B --device cuda:2 --answer evaluation_results/TDIUC/TDIUC_MGM-7B.jsonl
python inference_code/VQDv1_evaluation.py --model MGM-7B --device cuda:3 --answer evaluation_results/VQDv1/VQDv1_MGM-7B.jsonl

# MGM-7B-HD
python inference_code/DVQA_evaluations.py --model MGM-7B-HD --device cuda:0 --answer evaluation_results/DVQA/dvqa_MGM-7B-HD.jsonl
python inference_code/tallyqa_evaluations.py --model MGM-7B-HD --device cuda:1 --answer evaluation_results/TallyQA/TallyQA_MGM-7B-HD.jsonl
python inference_code/TDIUC_evaluations.py --model MGM-7B-HD --device cuda:2 --answer evaluation_results/TDIUC/TDIUC_MGM-7B-HD.jsonl
python inference_code/VQDv1_evaluation.py --model MGM-7B-HD --device cuda:3 --answer evaluation_results/VQDv1/VQDv1_MGM-7B-HD.jsonl

# Evaluation for MGM-7B
python eval_script/DVQA_eval.py --path evaluation_results/DVQA/dvqa_MGM-7B.jsonl
python eval_script/TallyQA_eval.py --path evaluation_results/TallyQA/TallyQA_MGM-7B.jsonl
python eval_script/TDIUC_eval.py --path evaluation_results/TDIUC/TDIUC_MGM-7B.jsonl
python eval_script/VQDv1_eval.py --path evaluation_results/VQDv1/VQDv1_MGM-7B.jsonl

# Evaluation for MGM-7B-HD
python eval_script/DVQA_eval.py --path evaluation_results/DVQA/dvqa_MGM-7B-HD.jsonl
python eval_script/TallyQA_eval.py --path evaluation_results/TallyQA/TallyQA_MGM-7B-HD.jsonl
python eval_script/TDIUC_eval.py --path evaluation_results/TDIUC/TDIUC_MGM-7B-HD.jsonl
python eval_script/VQDv1_eval.py --path evaluation_results/VQDv1/VQDv1_MGM-7B-HD.jsonl