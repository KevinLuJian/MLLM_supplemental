from openai import OpenAI
import os
import json
import argparse
import base64
from lavis.datasets.builders import load_dataset
from io import BytesIO

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def eval_model(args):  
    client = OpenAI()

    ans_file = open(args.answers_file, "a")

    data = load_dataset("vqdv1_dataset")
    i = 0
    import time
    start_time = time.time()
    for line in data['val']:
        print(line.keys())
        question = line["question"]
      
        prompt = "Generate a list of bounding box coordinates around the objects mentioned in the prompt if they exist in the image. Even if the prompt uses a singular verb like `is', generate multiple bounding boxes if multiple objects satisfy the query. The bounding box list should be formatted as: [[x_min, y_min, x_max, y_max]], and it can contain zero or more bounding boxes. Only provide the bounding box list, without any additional descriptions."
        questions = f"query: {line['question']}, instruction: {prompt}"

        image_id = line['image_path']
        image = line['image']
        base64_image = encode_image(image)
        

        while True:
            try:
                response = client.chat.completions.create(
                    model= args.model,
                    messages=[
                    {
                        "role": "user",
                        "content": [
                        {"type": "text", "text": questions},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        ],
                    }
                    ],
                max_tokens=200,
                )
                break
            except Exception as e:
                print(f"error: {e}, retrying...")
                continue

        predicted_answer = response.choices[0].message.content
        labeled_answer = line["answer"]
        question_type = line['question_type']
        width = line['width']
        height = line['height']

        record ={
            "question_id": i,
            "question": question,
            "image_id": image_id,
            "labeled_answer": labeled_answer,
            "predicted_answer": predicted_answer,
            "model_id": args.model,
            "question_type": question_type,
            "width": width,
            "height": height,
            "prompt": prompt,
        }

        ans_file.write(json.dumps(record) + "\n")
        ans_file.flush()
        print(f"question{i}: question:{question}, predicted_answer: {predicted_answer}, labeled_answer: {labeled_answer}")
        
        i += 1

    ans_file.close()
    end_time = time.time()
    running_time = end_time - start_time
    print(f"running time: {running_time}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o") # gpt-4-turbo
    parser.add_argument("--answers-file", type=str, default="Testing_result/GPT4V/VQDv1/a.json")

    args = parser.parse_args()
    args.answers_file = f"VQDv1_GPT4_{args.model}.jsonl"
    print(args.answers_file)
    eval_model(args)
