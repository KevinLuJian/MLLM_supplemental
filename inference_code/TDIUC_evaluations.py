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
elif args.model == 'llava-OV':
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
elif args.model == 'MGM-7B' or args.model == 'MGM-7B-HD':
    from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from mgm.conversation import conv_templates, SeparatorStyle
    from mgm.model.builder import load_pretrained_model
    from mgm.utils import disable_torch_init
    from mgm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    model_path = "work_dirs/MGM/MGM-7B" if args.model == 'MGM-7B' else "work_dirs/MGM/MGM-7B-HD"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(\
                model_path, None, model_name, device=args.device)

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
    if args.model == 'MGM-7B' or args.model == 'MGM-7B-HD':
        conv_mode = "vicuna_v1"
        conv = conv_templates[conv_mode].copy()
        inp = get_prompt(question, prompts)
        image = [image]

        if hasattr(model.config, 'image_size_aux'):
            if not hasattr(image_processor, 'image_size_raw'):
                image_processor.image_size_raw = image_processor.crop_size.copy()
            image_processor.crop_size['height'] = model.config.image_size_aux
            image_processor.crop_size['width'] = model.config.image_size_aux
            image_processor.size['shortest_edge'] = model.config.image_size_aux
        
        image_tensor = process_images(image, image_processor, model.config)
    
        image_grid = getattr(model.config, 'image_grid', 1)
        if hasattr(model.config, 'image_size_aux'):
            raw_shape = [image_processor.image_size_raw['height'] * image_grid,
                        image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor 
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                        size=raw_shape,
                                                        mode='bilinear',
                                                        align_corners=False)
        else:
            image_tensor_aux = []

        if image_grid >= 2:            
            raw_image = image_tensor.reshape(3, 
                                            image_grid,
                                            image_processor.image_size_raw['height'],
                                            image_grid,
                                            image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(-1, 3,
                                        image_processor.image_size_raw['height'],
                                        image_processor.image_size_raw['width'])
                    
            if getattr(model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(global_image, 
                                                            size=[image_processor.image_size_raw['height'],
                                                                    image_processor.image_size_raw['width']], 
                                                            mode='bilinear', 
                                                            align_corners=False)
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)
    
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(model.device, dtype=torch.float16) for image in image_tensor_aux]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(model.device, dtype=torch.float16)

        if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = (DEFAULT_IMAGE_TOKEN + '\n')*len(image) + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # add image split string
        if prompt.count(DEFAULT_IMAGE_TOKEN) >= 2:
            final_str = ''
            sent_split = prompt.split(DEFAULT_IMAGE_TOKEN)
            for _idx, _sub_sent in enumerate(sent_split):
                if _idx == len(sent_split) - 1:
                    final_str = final_str + _sub_sent
                else:
                    final_str = final_str + _sub_sent + f'Image {_idx+1}:' + DEFAULT_IMAGE_TOKEN
            prompt = final_str

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        return (input_ids, image_tensor, image_tensor_aux)

    elif args.model == 'CogVLM':
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
    elif args.model == 'llava-OV':
        prompts = get_prompt(question, prompts)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prompts}"},
                ],
            },
        ]
        prompts = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompts, return_tensors="pt").to(model.device, torch.float16)
        return inputs
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
            elif args.model == 'MGM-7B' or args.model == 'MGM-7B-HD':
                input_ids, image_tensor, image_tensor_aux = inputs
                outputs = model.generate(
                        input_ids,
                        images=image_tensor,
                        images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
                        do_sample=False,
                        bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                        eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                        pad_token_id=tokenizer.pad_token_id,  # Pad token
                        temperature=0,
                        repetition_penalty=1.2,
                        length_penalty=1.5,
                        max_new_tokens=100,
                        num_beams=1,
                        top_p=0.9)
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
        elif args.model == 'MGM-7B' or args.model == 'MGM-7B-HD':
            answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
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