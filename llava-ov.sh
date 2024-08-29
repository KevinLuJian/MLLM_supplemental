python inference_code/DVQA_evaluations.py --model llava-OV --device cuda:0 --answer evaluation_results/DVQA/dvqa_llava-OV.jsonl
python inference_code/tallyqa_evaluations.py --model llava-OV --device cuda:1 --answer evaluation_results/TallyQA/TallyQA_llava-OV.jsonl
python inference_code/TDIUC_evaluations.py --model llava-OV --device cuda:2 --answer evaluation_results/TDIUC/TDIUC_llava-OV.jsonl
python inference_code/VQDv1_evaluation.py --model llava-OV --device cuda:3 --answer evaluation_results/VQDv1/VQDv1_llava-OV.jsonl

# Evaluation for llava-OV
python eval_script/DVQA_eval.py --path evaluation_results/DVQA/dvqa_llava-OV_processed.jsonl
python eval_script/TallyQA_eval.py --path evaluation_results/TallyQA/TallyQA_llava-OV_processed.jsonl
python eval_script/TDIUC_eval.py --path evaluation_results/TDIUC/TDIUC_llava-OV_processed.jsonl
python eval_script/VQDv1_eval.py --path evaluation_results/VQDv1/VQDv1_llava-OV.jsonl