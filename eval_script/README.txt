To evaluate the models on different datasets(TallyQA, DVQA, VQDv1, TDIUC). First prepare the answer file, in jsonl format.

to run the evaluations, follow the command below: 
python DVQA_eval.py --path /path/to/DVQA_answer.jsonl


python TallyQA_eval.py --path /path/to/TallyQA_answer.jsonl --mode micro \\This evalaute the micro accuracy

python TallyQA_eval.py --path /path/to/TallyQA_answer.jsonl --mode macro \\This evalaute the macro accuracy


python TDIUC_eval.py --path /path/to/TDIUC_answer.jsonl


python VQDv1_eval.py --path /path/to/VQDv1_answer.jsonl


