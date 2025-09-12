import pandas as pd
import ollama
import time
import os

import create_prompts as c_p

chunking = "1024_20"
only_text = True
models = ["llama3.1:70b", "llama4:scout"]
model = models[0]
ai_prompt = True

model_path = f"../data/dfs/{'only_text_' if only_text else ''}{chunking}/{model.replace(':','.')}/{'AI_prompt/' if ai_prompt else ''}"
# Create the directory if it doesn't exist
if not os.path.exists(model_path):
    os.makedirs(model_path)

def send_prompt(prompt, model):
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}], options={"temperature": 0})
    return response['message']['content']

def prompting_model(df, model, save_intermediate_results=False, save_every=10, ids_not_to_prompt=[]):
    print(f"Prompting model: {model}", flush=True)

    # Create a new column in the dataframe to store the responses
    if 'Model Classification' not in df.columns:
        df['Model Classification'] = None

    # Iterate through the dataframe
    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            if pd.notna(row['Model Classification']):
                print(f"Already processed: " + row['Reference Article ID'], flush=True)
                continue

            if len(ids_not_to_prompt) != 0 and row['Reference Article ID'] in ids_not_to_prompt:
                continue

            start_time = time.time()
            print(f"Processing: " + row['Reference Article ID'], flush=True)

            # Create the prompt
            if ai_prompt:
                prompt = c_p.create_prompt_ai_improved(row)
            else:
                prompt = c_p.create_prompt(row)
            
            # Send the prompt and get the response
            response = send_prompt(prompt, model)
            
            # Save the response to the new column
            df.at[index, 'Model Classification'] = response

            if save_intermediate_results and index % save_every == 0:
                df.to_pickle(f"{model_path}ReferenceErrorDetection_data_with_prompt_results_intermed.pkl")
            end_time = time.time()
            print(f"Took {round(end_time - start_time, 2)} seconds", flush=True)
            print("==================================", flush=True)
    return df

path = f"../data/dfs/{'only_text_' if only_text else ''}{chunking}/ReferenceErrorDetection_data_with_chunk_info.pkl"
print(path)

# read the dataframe from a pickle file
df = pd.read_pickle(path)

# df2_old = pd.read_pickle(f"../data/dfs/{embedding}{'_no_prev_chunking' if no_prev_chunking else ''}/{grobid_model}/ReferenceErrorDetection_data_with_prompt_results_{model}_intermed.pkl")
# ids_not_to_prompt = df2_old[df2_old['Model Classification'].notna()]['Reference Article ID'].tolist()
# print(ids_not_to_prompt)

print("Start prompting script", flush=True)
df2 = prompting_model(df, model, save_intermediate_results=True, save_every=5)

df2.to_pickle(f"{model_path}ReferenceErrorDetection_data_with_prompt_results.pkl")
df2.to_excel(f"{model_path}ReferenceErrorDetection_data_with_prompt_results.xlsx", index=False)