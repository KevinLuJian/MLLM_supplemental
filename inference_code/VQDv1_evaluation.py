from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
import torch
from PIL import Image, ImageOps
import re
import json
import time
import argparse
import os
from tqdm import tqdm

# Load processor and model
parser = argparse.ArgumentParser(description='Select model to use for visual question answering.')
parser.add_argument('--model', type=str, required=True, help='Model to use for visual question answering')
parser.add_argument('--device', type=str)
parser.add_argument('--answer', type=str)
args = parser.parse_args()


if args.model == 'LlavaNext':
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'LLaVA1.57b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype=torch.float16, low_cpu_mem_usage=True).to(0)
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
elif args.model == 'LLaVA1.513b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'QwenVL':
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig
    processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", torch_dtype=torch.float16, trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
elif args.model == 'CogVLM':
    from transformers import AutoModelForCausalLM, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(args.device).eval()

model.to(args.device)


ans_file = open(args.answer, 'a')



def get_prompt(question):
    if args.model == 'LlavaNext':
        prompt = 'Please generate a list of bounding boxes coordinates of the region this query describes. Use the format [[x\_min,y\_min,x\_max,y\_max]....]. Do not respond in sentences, and only generate the bounding boxes. If no such region exists in the image, respond with an empty list.'
        return f"[INST] <image>Query: \n{question}, prompt: {prompt}[/INST]"
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        prompt = 'Please answer the query by generating a list of bounding box coordinates around the objects the query is asking, and if no such object exists in the image, answer: [[]]'
        return f"USER: <image>\Query: {question},Instruction: {prompt}\nASSISTANT:"
    elif args.model == 'QwenVL':
        prompt = 'Detect and generate a bounding box for each instance of the requested object(s) within the image. If there are multiple instances of the object(s), ensure a separate bounding box is created for every single one, without exception. If the requested object(s) are not present, return no bounding boxes. Every matching instance must be framed individually to capture all occurrences accurately.'
        return f"{question}\n{prompt}"
    elif args.model == 'CogVLM':
        prompt = 'Please generate a list of bounding boxes coordinates of the region this query describes. Use the format [[x\_min,y\_min,x\_max,y\_max]....]. Do not respond in sentences, and only generate the bounding boxes. If no such region exists in the image, respond with an empty list.'
        return f"{question}\n{prompt}"

def extract_non_inst_text(text):
    non_inst_text = ''
    if args.model == 'LlavaNext':
        non_inst_text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    elif args.model == 'LLaVA1.513b' or args.model == 'LLaVA1.57b':
        match = re.search(r'ASSISTANT:(.*)', text, flags=re.DOTALL)
        if match:
            non_inst_text = match.group(1).strip()
    else:
        return text
    return non_inst_text

def get_input(question, image, image_path=None):
    if args.model == 'CogVLM':
        prompts = get_prompt(question)
        inputs = model.build_conversation_input_ids(\
                tokenizer, query=prompts, history=[], images=[image])
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(model.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(model.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(model.device),
            'images': [[inputs['images'][0].to(model.device).to(torch.bfloat16)]],
        }
        return inputs
    elif args.model == 'QwenVL':
        prompts = get_prompt(question)
        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': f"{prompts}"},
            ])
        return query
    else:
        prompts = get_prompt(question, prompts)
        inputs = processor(images=image, text=prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        return inputs

i = 0
start_time = time.time()

from lavis.datasets.builders import load_dataset
dataset = load_dataset("vqdv1_dataset")

with tqdm(total=len(dataset['val'])) as pbar:
    for a in dataset['val']:
        # print(dataset.keys())
        image = a['image']
        question = a['question']
        labeled_answer = a['answer']
        question_type = a['question_type']
        question_id = a['question_id']
        width = a['width']
        height = a['height']
        image_name = a['image_path']
        image_path = f"LAVIS/cache/VQDv1/images/val2014/{image_name}"

        inputs = get_input(question, image, image_path)        

        with torch.inference_mode():
            if args.model == 'QwenVL':
                outputs = model.chat(tokenizer, query=inputs, history=None,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=100, 
                    top_p=0.9,
                    repetition_penalty=1.2,
                    length_penalty=1.5,
                    temperature=0)
            else:
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=100,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    length_penalty=1.5,
                    temperature=0,
                )

        if args.model == 'QwenVL':
            answer = outputs[0]
        elif args.model == 'CogVLM':
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            answer = processor.decode(outputs[0], skip_special_tokens=True).strip()

        record = {
            "question_index": i,
            "question": question,
            "question_id": question_id,
            "image_id": image_name,
            "predicted_answer": answer,
            "labeled_answer": labeled_answer,
            "question_type": question_type,
            "model_id": args.model,
            "width":width,
            "height":height,
            "prompt": get_prompt(question)
        }
        ans_file.write(json.dumps(record) + "\n")
        ans_file.flush()
        print(f"predicted = {answer}, labeled = {labeled_answer}")
        i += 1

        # Update the progress bar
        pbar.update(1)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
ans_file.close()

