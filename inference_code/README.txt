To run the inference code:
example command:
python DVQA_evaluation.py --model [model_name], --device [cuda or mps] --answer [path to store the answer_file]

eg: python DVQA_evaluation.py --model LlavNext --device cuda --answer answer_file.jsonl
In our evluations, model options include:LlavaNext,InstructBlip,BLIP2,LLaVA1.513b, LaVA1.57b

