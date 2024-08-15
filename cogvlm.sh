python inference_code/DVQA_evaluations.py --model CogVLM --device cuda:0 --answer evaluation_results/DVQA/dvqa_CogVLM.jsonl
python inference_code/tallyqa_evaluations.py --model CogVLM --device cuda:1 --answer evaluation_results/TallyQA/TallyQA_CogVLM.jsonl
python inference_code/TDIUC_evaluations.py --model CogVLM --device cuda:2 --answer evaluation_results/TDIUC/TDIUC_CogVLM.jsonl
python inference_code/VQDv1_evaluation.py --model CogVLM --device cuda:3 --answer evaluation_results/VQDv1/VQDv1_CogVLM.jsonl
