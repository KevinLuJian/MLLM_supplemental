from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
import torch
from PIL import Image, ImageOps
import re
import json
import time
import argparse
import os

# Load processor and model


parser = argparse.ArgumentParser(description='Select model to use for visual question answering.')
parser.add_argument('--model', type=str, required=True, help='Model to use for visual question answering')
parser.add_argument('start', type=int)
args = parser.parse_args()


if args.model == 'LlavaNext':
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'InstructBlip':
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl",torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
elif args.model == 'BLIP2':
    from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'LLaVA1.57b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
elif args.model == 'LLaVA1.513b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

# Load dataset and prepare output file



def get_prompt(question, prompt):
    if args.model == 'LlavaNext':
        return f"[INST] <image>\n{question}, {prompt}/INST]"
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        return f"USER: <image>\n {question}, {prompt} \nASSISTANT"
    else:
        return f"{question}, {prompt}"
    
def extract_non_inst_text(text):
    # This regex finds all text outside of [INST]...[/INST] blocks
    non_inst_text = ''
    if args.model == 'LlavaNext' or args.model == 'BLIP2' or args.model == 'InstructBlip':
        non_inst_text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    else:
        match = re.search(r'ASSISTANT:(.*)', text, flags=re.DOTALL)
        if match:
            non_inst_text = match.group(1).strip()

    return non_inst_text.strip()

# Process data one by one
from lavis.datasets.builders import load_dataset
TDIUC_dataset = load_dataset("TDIUC_dataset")
ans_file = open(f'path/to/answer.jsonl', "a")
i = 0
start_time = time.time()

for a in TDIUC_dataset['test']:
    if i < args.start:
        i += 1
        continue

    question = a['question']
    labeled_answer = a['answer']
    question_type = a['question_type']
    image_id = a['image_name']
    image = a['image']
    question_id = a['question_id']

    prompt = "Please answer the question in one word, answer 'doesnotapply' if you believe the question is not related to the image, or cannot be answered."
    prompt = get_prompt(question, prompt)

    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True, truncation=True).to("cuda:0")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=50,
            top_p=0.9,
            repetition_penalty=1.2,
            length_penalty=1.5,
            temperature=0,
        )

    answer = processor.decode(outputs[0], skip_special_tokens=True).strip()

    record = {
        "question_index": i,
        "question": question,
        "question_id": question_id,
        "image_id": image_id,
        "predicted_answer": answer,
        "labeled_answer": labeled_answer,
        "question_type": question_type,
        "model_id": args.model,
    }
    ans_file.write(json.dumps(record) + "\n")
    ans_file.flush()
    
    print(f"{args.model}, {i} questions have been processed, question:{question}, predicted_answer:{answer}, label: {labeled_answer}")
    i += 1

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
ans_file.close()