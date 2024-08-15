from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
import torch
from PIL import Image, ImageOps
import re
import json
import time
import argparse
import os
from tqdm import tqdm

DEFAULT_PROMPT =  "Please answer the question in one word, answer 'doesnotapply' if you believe the question is not related to the image, or cannot be answered."


# Load processor and model
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

parser = argparse.ArgumentParser(description='Select model to use for visual question answering.')
parser.add_argument('--model', type=str, required=True, help='Model to use for visual question answering')
parser.add_argument('--device', type=str)
parser.add_argument('--answer', type=str)
args = parser.parse_args()


if args.model == 'LlavaNext':
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'InstructBlip':
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl",torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
elif args.model == 'BLIP2':
    from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'LLaVA1.57b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained(
        'llava-hf/llava-1.5-7b-hf', 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    )
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
elif args.model == 'LLaVA1.513b':
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'paligemma-3b':
    from transformers import AutoProcessor, AutoModelForPreTraining,PaliGemmaForConditionalGeneration
    processor = AutoProcessor.from_pretrained("/gpfs/fs2/scratch/jlu59/home/Llava_Next/paligemma-3b-pt-224")
    model = PaliGemmaForConditionalGeneration.from_pretrained("/gpfs/fs2/scratch/jlu59/home/Llava_Next/paligemma-3b-pt-224")
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

# Load dataset and prepare output file
ans_file = open(args.answer, "a")


def get_prompt(question, prompt = ''):
    if args.model == 'BLIP2':
        return f"Question:{question},{prompt} Answer:"
    elif args.model == 'LlavaNext':
        return f"[INST] <image>\n{question}, {prompt}[/INST]"
    elif args.model == 'InstructBlip':
        return f"[INST]{question}, {prompt}[/INST]"
    elif args.model == 'LLaVA1.57b' or args.model == 'LLaVA1.513b':
        return f"USER: <image>\n{question},{prompt}\nASSISTANT:"
    else:
        return f"{question}, {prompt}"

# def extract_non_inst_text(text):
#     # This regex finds all text outside of [INST]...[/INST] blocks
#     non_inst_text = ''
#     if args.model == 'LlavaNext' or args.model == 'BLIP2' or args.model == 'InstructBlip':
#         non_inst_text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
#     else:
#         match = re.search(r'ASSISTANT:(.*)', text, flags=re.DOTALL)
#         if match:
#             non_inst_text = match.group(1).strip()

#     return non_inst_text.strip()

def extract_answer(answer):
    if args.model == 'LlavaNext':
        print('LlavaNext extract')
        match = re.search(r"INST\](.*)$", answer)
        # Check if the match was found
        if match:
            extracted_text = match.group(1).strip()
            # print("Extracted text:", extracted_text)
            return extracted_text
        else:
            return ''
    elif args.model == 'LLaVA1.57b' or args.model == 'LLaVA1.513b':
        match = re.search(r'ASSISTANT:(.*)', answer, flags=re.DOTALL)
        if match:
            non_inst_text = match.group(1).strip()
            return non_inst_text
    else:
        return answer

def get_input(question, prompts, image, image_path=None):
    if args.model == 'CogVLM':
        prompts = get_prompt(question, prompts)
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
        prompts = get_prompt(question, prompts)
        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': f"{prompts}"},
            ])
        return query
    else:
        prompts = get_prompt(question, prompts)
        inputs = processor(images=image, text=prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        return inputs

# Process data one by one
from lavis.datasets.builders import load_dataset
TDIUC_dataset = load_dataset("TDIUC_dataset")
i = 0
start_time = time.time()


# Initialize tqdm with the length of your dataset
with tqdm(total=len(TDIUC_dataset['test'])) as pbar:
    for a in TDIUC_dataset['test']:
        question = a['question']
        labeled_answer = a['answer']
        question_type = a['question_type']
        image_id = a['image_name']
        image = a['image']
        question_id = a['question_id']
        image_path = f"LAVIS/cache/TDIUC/images/{image_id}"

        prompt = DEFAULT_PROMPT
        inputs = get_input(question, prompt, image, image_path)

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

        answer = extract_answer(answer)
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
        
        print(f"predicted = {answer}, labeled = {labeled_answer}")
        i += 1

        # Update the progress bar
        pbar.update(1)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
ans_file.close()