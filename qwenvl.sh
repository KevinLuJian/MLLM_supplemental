python inference_code/DVQA_evaluations.py --model QwenVL --device cuda:0 --answer evaluation_results/DVQA/dvqa_QwenVL.jsonl
python inference_code/tallyqa_evaluations.py --model QwenVL --device cuda:1 --answer evaluation_results/TallyQA/TallyQA_QwenVL.jsonl
python inference_code/TDIUC_evaluations.py --model QwenVL --device cuda:2 --answer evaluation_results/TDIUC/TDIUC_QwenVL.jsonl
python inference_code/VQDv1_evaluation.py --model QwenVL --device cuda:3 --answer evaluation_results/VQDv1/VQDv1_QwenVL.jsonl


# Evaluation
python eval_script/DVQA_eval.py --path evaluation_results/DVQA/dvqa_QwenVL.jsonl
python eval_script/TallyQA_eval.py --path evaluation_results/TallyQA/TallyQA_QwenVL.jsonl
python eval_script/TDIUC_eval.py --path evaluation_results/TDIUC/TDIUC_QwenVL.jsonl
python eval_script/VQDv1_eval.py --path evaluation_results/VQDv1/VQDv1_QwenVL.jsonl --model qwen