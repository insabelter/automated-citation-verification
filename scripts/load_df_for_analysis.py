import pandas as pd
import json

def sort_df(df):
    df = df.sort_values(by=['Citing Article ID', 'Reference Article ID'], ascending=[True, True]).reset_index(drop=True)
    return df

def load_df(model_type, embedding, no_prev_chunking, gpt_model, batched, annotated=False, corrected_statements=False, two_labels=False):
    path = f"../data/dfs{'/annotated_data' if annotated else ''}{'/two_labels' if two_labels else ''}{'/corrected_statements' if corrected_statements else ''}/{embedding}{'_no_prev_chunking' if no_prev_chunking else ''}/{model_type}/ReferenceErrorDetection_data_with_prompt_results{'_batched' if batched else ''}{'_'+gpt_model.replace(':','.') if gpt_model != 'gpt-3.5-turbo-0125' else ''}.pkl"
    df = pd.read_pickle(path)
    return df

def remove_json_colons(json_text):
    if json_text and '{' in json_text and '}' in json_text:
        json_text = json_text[json_text.find('{'):json_text.rfind('}') + 1]
    return json_text

# Add extra columns for the model classification label and explanation by extracting the information from the JSON
# If the JSON is misformed due to leading ```json and trailing ``` then remove them
# Make sure that correct label and model label are both lower case and do not end with d (unsubstaniate instead of unsubstantiated)
def reshape_model_classification(df):
    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            try:
                row['Model Classification'] = remove_json_colons(row['Model Classification'])
                model_classification = json.loads(row['Model Classification'])
                label = model_classification['label'].lower()
                df.at[row.name, 'Model Classification Label'] = label if not label.endswith('d') else label[:-1]
                df.at[row.name, 'Model Classification Explanation'] = model_classification['explanation']
            except json.JSONDecodeError as e:
                print(f"Row {index} Model Classification could not be decoded: {e}")
                print(row['Model Classification'])
                df.at[row.name, 'Model Classification Label'] = None
                df.at[row.name, 'Model Classification Explanation'] = None
        else:
            df.at[row.name, 'Model Classification Label'] = None
            df.at[row.name, 'Model Classification Explanation'] = None
        df.at[row.name, 'Label'] = df.at[row.name, 'Label'].lower()
    return df

def load_df_for_analysis(model_type, embedding, no_prev_chunking, gpt_model, batched, annotated=False, corrected_statements=False, two_labels=False, remove_added=True):
    df = load_df(model_type, embedding, no_prev_chunking, gpt_model, batched, annotated, corrected_statements, two_labels)
    if 'Suited for Task' in df.columns and remove_added:
        df = df[df['Suited for Task'] != "Added"]
    df = sort_df(df)
    df = reshape_model_classification(df)
    return df