import pandas as pd
import os
import json
import time
from openai import OpenAI

import create_prompts as c_p

models = ["gpt-3.5-turbo-0125", "gpt-4.1-nano-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]
model = models[3]

chunking = "1024_20"
only_text = True
ai_prompt = False
suit_prompt = True
fixed_coverage = True
indications = 'any' # 'strong', 'any', None

path = f"../data/dfs/{'only_text_' if only_text else ''}{chunking}/ReferenceErrorDetection_data_with_chunk_info.pkl"

# read the dataframe from a pickle file
df = pd.read_pickle(path)

folder_path = f"{'only_text_' if only_text else ''}{chunking}/{model}{'/AI_prompt/' if ai_prompt else ''}{'/suit_' if suit_prompt else ''}{'strong_ind_' if indications == 'strong' else 'any_ind_' if indications == 'any' else 'none_ind_' if (indications == None and suit_prompt) else ''}{'fixed_cov' if fixed_coverage else ''}{'/' if suit_prompt else ''}"

os.makedirs(f"../data/batch_responses/{folder_path}", exist_ok=True)
responses_dict_path = f"../data/batch_responses/{folder_path}/{model}_responses_dict_batch.json"

with open('../open_ai_key.txt', 'r') as file:
    open_ai_key = file.read().strip()

responses_dict = {}
ids_to_ignore = []
prompt_chars = []

def load_responses_dict():
    global responses_dict
    global ids_to_ignore

    # Load existing responses if the file exists
    try:
        with open(responses_dict_path, 'r') as file:
            responses_dict = json.load(file)
        ids_to_ignore = [int(key) for key in responses_dict.keys()]
    except FileNotFoundError:
        ids_to_ignore = []

def check_batch(batch_id, client):
    batch = client.batches.retrieve(batch_id)
    print(f"{batch_id} - Current status: {batch.status}", flush=True)
    if (batch.status == 'in_progress'):
        print(f"{batch.request_counts.completed} / {batch.request_counts.total} completed", flush=True)
        current_time = time.time()
        print(f"Current time and date: {time.ctime(current_time)}", flush=True)

    if batch.status == 'completed' or batch.status == 'failed':
        return batch
    return None

def wait_for_batch_completion(batch_id, client, interval=10):
    while True:
        batch = check_batch(batch_id, client)
        if batch != None:
            return batch
        time.sleep(interval)

def save_completed_batches(batches):
    global responses_dict

    # save responds of completed batches
    for batch in batches:
        if batch.status != "completed":
            continue
        model_responses = client.files.content(batch.output_file_id).text

        # Parse the model_responses into a list of objects
        responses_list = [json.loads(line) for line in model_responses.splitlines()]
        # print(responses_list)

        try:
            for response in responses_list:
                responses_dict[int(response['custom_id'].split('-')[1])] = response
                responses_dict = dict(sorted(responses_dict.items(), key=lambda item: int(item[0])))
        except NameError:
            responses_dict = {int(response['custom_id'].split('-')[1]): response for response in responses_list}
    # Save responses_dict to a JSON file
    with open(responses_dict_path, 'w') as file:
        json.dump(responses_dict, file, indent=4)
    print(f"Saved {len(responses_dict)} responses to {responses_dict_path}", flush=True)

def wait_for_in_progress_batch_completions(client, interval=60):
    print("Waiting for in-progress batches...", flush=True)

    current_millis = int(time.time())
    recently = current_millis - 24 * 60 * 60

    open_batches = client.batches.list()
    relevant_open_batches = [batch for batch in open_batches if batch.created_at >= recently]
    in_progress_batch_ids = [batch.id for batch in relevant_open_batches if batch.status == 'in_progress']

    completed_batches = []
    for batch_id in in_progress_batch_ids:
        print("Waiting for batch to complete: " + batch_id, flush=True)
        current_time = time.time()
        batch = wait_for_batch_completion(batch_id, client, interval)
        print(f"Batch {batch_id} completed in {time.time() - current_time} seconds", flush=True)
        if batch:
            completed_batches.append(batch)

    save_completed_batches(completed_batches)

def create_batch_files(df, model, number_files=1, ignore_ids=[], ai_prompt=False, suit_prompt=False, indications=None, fixed_coverage=False):
    global prompt_chars

    output_dir = f"../data/batch_files/{folder_path}"
    # Empty the folder if it exists
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    os.makedirs(output_dir, exist_ok=True)

    output_files = []
    for i in range(number_files):
        output_file = os.path.join(output_dir, f"prompt_batch_{i}.jsonl")
        # If the file already exists, empty it
        open(output_file, "w").close()
        output_files.append(output_file)
    
    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes' and index not in ignore_ids:
            if ai_prompt:
                prompt = c_p.create_prompt_ai_improved(row)
            elif suit_prompt:
                prompt = c_p.create_prompt_suit(row, fixed_coverage=fixed_coverage, indications=indications)
            else:
                prompt = c_p.create_prompt(row)

            prompt_char = len(prompt)
            prompt_chars.append(prompt_char)

            json_sequence = {
                "custom_id": f"request-{index}", 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": model, 
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0,
                }
            }

            output_file = output_files[index % number_files]
            with open(output_file, "a") as f:
                f.write(json.dumps(json_sequence) + "\n")
                
    # Remove empty output files from list
    output_files = [file for file in output_files if os.path.getsize(file) > 0]
    
    return output_files

load_responses_dict()

client = OpenAI(api_key=open_ai_key)
# wait_for_in_progress_batch_completions(client, 120)

load_responses_dict()

batch_file_paths = create_batch_files(df, model, 5, ids_to_ignore, ai_prompt=ai_prompt, suit_prompt=suit_prompt, indications=indications, fixed_coverage=fixed_coverage)
    
batch_input_files = []
batches = []

def prompt_model_in_batches(interval=60):
    global batch_input_files
    global batches

    for batch_file_path in batch_file_paths:
        # Creating input file
        if os.stat(batch_file_path).st_size == 0:
            print(f"Skipping empty file: {batch_file_path}", flush=True)
            continue
        batch_input_file = client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )
        print(batch_input_file, flush=True)
        batch_input_files.append(batch_input_file)

        # Starting batch job
        batch_input_file_id = batch_input_file.id
        batch_creation_response = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print("Started: " + batch_creation_response.id, flush=True)

        start_time = time.time()
        time.sleep(5)
        # Check the status of the created batch until it is completed
        while True:
            batch_id = batch_creation_response.id
            batch = check_batch(batch_id, client)
            if batch:
                if batch.status == "failed":
                    print(f"Batch {batch.id} failed", flush=True)
                    return
                elif batch.status == "completed":
                    batches.append(batch)
                    print(f"Batch {batch.id} completed in {time.time() - start_time} seconds", flush=True)
                    break
            time.sleep(interval)
        
    save_completed_batches(batches)

prompt_model_in_batches(30)