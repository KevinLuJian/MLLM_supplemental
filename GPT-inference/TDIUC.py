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

    i = 0
    import time
    tdiuc = load_dataset("TDIUC_dataset")
    start_time = time.time()
    for line in tdiuc['test']:
        # print(line.keys())
        question = line["question"]
        prompt = "please answer in one word, answer 'doesnotapply' if you believe the question is not related to the image, or cannot be answered."
        question_id = line["question_id"]
        image = line['image']

        base64_image = encode_image(image)

        while True:
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{question}, {prompt}"},
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
        question_type = line["question_type"]
        record = {
            "question_index": i,
            "question_id": question_id,
            "question": question,
            "predicted_answer": predicted_answer,
            "label_answer": labeled_answer,
            "question_type": question_type,
        }
        ans_file.write(json.dumps(record) + "\n")
        ans_file.flush()
        print(f"question{i}: question:{question}, predicted_answer: {predicted_answer}, labeled_answer: {labeled_answer}")
        
        i += 1

    ans_file.close()
    end_time = time.time()
    running_time = end_time - start_time
    print(f"running time: {running_time}")

    # print(response.choices[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o") #gpt-4-turbo
    parser.add_argument("--answers-file", type=str, default="GPT4V/TallyQA/TallyQA_4v_10000.json")
    args = parser.parse_args()
    args.answers_file = f"TDIUC_{args.model}.jsonl"
    print(f"store the result in {args.answers_file}")
    eval_model(args)
