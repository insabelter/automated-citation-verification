import pandas as pd
import ollama
import time

def format_excerpts(excerpt_list):
    excerpts_text = ""
    for id, excerpt in enumerate(excerpt_list):
        excerpts_text += f"Excerpt {id+1}: \n{excerpt}\n\n"
    return excerpts_text

def create_prompt(df_row):
    title = df_row['Citing Article Title']
    statement = df_row['Statement with Citation']
    reference_title = df_row['Reference Article Title']
    reference_abstract = df_row['Reference Article Abstract']
    reference_excerpts = format_excerpts(df_row['Top_3_Chunk_Texts'])

    prompt = f"""   
You are an experienced scientific writer and editor. 
You will be given a statement from an article that cites a reference article and information from the reference article. 
You will determine and explain if the reference article supports the statement.  
    
Specifically, choose a label from "Fully substantiate", "Partially substantiate", and "Unsubstantiate". 
Further explanations of the labels are as follows: 
"Fully substantiated": The reference article fully substantiates the relevant part of the statement from the present article. 
"Partially substantiated": According to the reference article, there is a minor error in the statement but the error does not invalidate the purpose of the statement. 
"Unsubstantiate": The reference part does not substantiate any part of the statement. This could be because the statement is contradictory to, unrelated to, or simply missing from the reference article.  
    
Format your answer in JSON with two elements: "label" and "explanation". 
Your explanation should be short and concise. 
    
# The citing article
Title: {title} 
Statement: {statement}
    
# The reference article 
Title: {reference_title} 
Abstract: {reference_abstract} 
Excerpts: \n\n{reference_excerpts}
"""

    return prompt

def send_prompt(prompt, model):
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
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
            prompt = create_prompt(row)
            
            # Send the prompt and get the response
            response = send_prompt(prompt, model)
            
            # Save the response to the new column
            df.at[index, 'Model Classification'] = response

            if save_intermediate_results and index % save_every == 0:
                df.to_pickle(f"../data/dfs/{embedding}{'_no_prev_chunking' if no_prev_chunking else ''}/{grobid_model}/ReferenceErrorDetection_data_with_prompt_results_{model.replace(':','.')}_intermed.pkl")
            end_time = time.time()
            print(f"Took {round(end_time - start_time, 2)} seconds", flush=True)
            print("==================================", flush=True)
    return df

embedding = "te3l" # / "te3s"
grobid_model = "full_model"
no_prev_chunking = True

path = f"../data/dfs/{embedding}{'_no_prev_chunking' if no_prev_chunking else ''}/{grobid_model}/ReferenceErrorDetection_data_with_chunk_info.pkl"

# read the dataframe from a pickle file
df = pd.read_pickle(path)

models = ["llama3.1:70b", "llama3.1:405b", "llama3.3"]
model = models[1]

df2_old = pd.read_pickle(f"../data/dfs/{embedding}{'_no_prev_chunking' if no_prev_chunking else ''}/{grobid_model}/ReferenceErrorDetection_data_with_prompt_results_{model}_intermed.pkl")
ids_not_to_prompt = df2_old[df2_old['Model Classification'].notna()]['Reference Article ID'].tolist()
print(ids_not_to_prompt)

print("Start prompting script", flush=True)
df2 = prompting_model(df2_old, model, save_intermediate_results=True, save_every=1, ids_not_to_prompt=ids_not_to_prompt)

df2.to_pickle(f"../data/dfs/{embedding}{'_no_prev_chunking' if no_prev_chunking else ''}/{grobid_model}/ReferenceErrorDetection_data_with_prompt_results_{model.replace(':','.')}.pkl")