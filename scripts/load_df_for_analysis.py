import pandas as pd
import json

def sort_df(df):
    df = df.sort_values(by=['Citing Article ID', 'Reference Article ID'], ascending=[True, True]).reset_index(drop=True)
    return df

def load_df(chunking, only_text, model):
    path = f"../data/dfs/{'only_text_' if only_text else ''}{chunking}/{model}/ReferenceErrorDetection_data_with_prompt_results.pkl"
    df = pd.read_pickle(path)
    return df

def remove_json_colons(json_text):
    if json_text and '{' in json_text and '}' in json_text:
        json_text = json_text[json_text.find('{'):json_text.rfind('}') + 1]
    return json_text

def find_label_within_non_json_text(text):
    if not 'label' in text.lower():
        return None
    if 'substantiated' in text.lower():
        return 'Substantiated'
    elif 'unsubstantiated' in text.lower():
        return 'Unsubstantiated'
    return None

# Add extra columns for the model classification label and explanation by extracting the information from the JSON
# If the JSON is misformed due to leading ```json and trailing ``` then remove them
# Make sure that correct label and model label are both lower case and do not end with d (unsubstaniate instead of unsubstantiated)
def reshape_model_classification(df):
    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            try:
                row['Model Classification'] = remove_json_colons(row['Model Classification'])
                model_classification = json.loads(row['Model Classification'])
                df.at[row.name, 'Model Classification Label'] = model_classification['label']
                df.at[row.name, 'Model Classification Explanation'] = model_classification['explanation']
            except (json.JSONDecodeError, KeyError) as e:
                label = find_label_within_non_json_text(row['Model Classification'])
                if label:
                    df.at[row.name, 'Model Classification Label'] = label
                    df.at[row.name, 'Model Classification Explanation'] = row['Model Classification']
                else:
                    print(f"Row {index} Model Classification could not be decoded: {e}")
                    print(row['Model Classification'])
                    df.at[row.name, 'Model Classification Label'] = None
                    df.at[row.name, 'Model Classification Explanation'] = None
        else:
            df.at[row.name, 'Model Classification Label'] = None
            df.at[row.name, 'Model Classification Explanation'] = None
        df.at[row.name, 'Label'] = df.at[row.name, 'Label']
    return df

def load_df_for_analysis(chunking, only_text, model):
    df = load_df(chunking, only_text, model)
    df = sort_df(df)
    df = reshape_model_classification(df)
    return df