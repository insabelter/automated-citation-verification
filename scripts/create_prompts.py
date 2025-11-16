import re

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def format_excerpts(excerpt_list):
    excerpts_text = ""
    for id, excerpt in enumerate(excerpt_list):
        excerpts_text += f"Excerpt {id+1}: \n{normalize_whitespace(excerpt)}\n\n"
    return excerpts_text

def create_prompt(df_row):
    title = df_row['Citing Article Title']

    statement = df_row["Corrected Statement"]
    assert statement is not None and statement != '', "Statement cannot be None or empty"

    reference_number = df_row['Reference Number']
    reference_title = df_row['Reference Article Title']
    reference_abstract = df_row['Reference Article Abstract']
    reference_excerpts = format_excerpts(df_row['Top_3_Chunk_Texts'])

    prompt = f"""   
You are an experienced scientific writer and editor. 
You will be given a citation statement from an article that cites a reference article.
From this same reference article you will receive the additional information, including the title, the abstract of the article and the top 3 most relevant excerpts from the reference article. The relevance of the excerpts was previously determined by another large language model based on the citation statement.
Your task is to determine and explain if the reference article supports the given citation statement.  

The statement sentence can contain multiple citations and can refer to multiple reference articles, which are all cited in IEEE style.
You are given the number of the reference article that you should check ("Reference Number"). When for example the statement is "X is true [37, 38] and Y is false under certain conditions [39]", and you are given the reference number 37, you should only check the first part of the statement that refers to the reference article 37.
    
As your classification result, decide between the two labels "Substantiated" and "Unsubstantiated". 
Further explanations of the labels are as follows: 
"Substantiated": The reference article fully substantiates the relevant part of the presented citation statement. This means that only based on the information from the reference article, the statement does not contain errors and can be considered correct. 
"Unsubstantiated": The reference article does not substantiate the relevant part of the presented citation statement. This could be because the statement is contradictory to, unrelated to, or simply missing from the reference article. All of these options would indicate that the citation is incorrect based on the cited references.
    
Format your answer in JSON with two elements: "label" and "explanation". 
Your explanation should be short and concise. 
    
# The citing article

-- Title: {title} 

-- Statement: {statement}
    
# The reference article 

-- Reference Number: {reference_number}

-- Title: {reference_title} 

-- Abstract: {reference_abstract} 

-- Excerpts: \n{reference_excerpts}
"""

    return prompt

def create_prompt_ai_improved(df_row):
    title = df_row['Citing Article Title']

    statement = df_row["Corrected Statement"]
    assert statement is not None and statement != '', "Statement cannot be None or empty"

    reference_number = df_row['Reference Number']
    reference_title = df_row['Reference Article Title']
    reference_abstract = df_row['Reference Article Abstract']
    reference_excerpts = format_excerpts(df_row['Top_3_Chunk_Texts'])

    prompt = f"""
You are an experienced scientific writer and editor. Your task is to evaluate whether a reference article supports a given citation statement from another article.

Inputs you will receive:
1. A citation statement from a citing article. This statement may include multiple IEEE-style citations (e.g., "[37, 38, 39]").
2. The reference article being evaluated, including:
   - The reference number (e.g., 37)
   - The title
   - The abstract
   - The top 3 most relevant excerpts (pre-selected by a language model based on the citation statement)

Your task:
Determine whether the relevant part of the citation statement is substantiated by the provided reference article.

Important rules:
- Only assess the portion of the citation statement that corresponds to the given reference number.
  For example, if the statement is:
  "X is true [37, 38] and Y is false under certain conditions [39]"
  and the reference number is 37, you should only evaluate whether the claim "X is true" is substantiated by reference 37.
- Use only the abstract and provided excerpts to make your decision. Do not assume additional content not given.

Classification labels:
- "Substantiated": The relevant part of the statement is fully supported by the reference article. It is consistent with, clearly stated in, or directly derived from the reference content.
- "Unsubstantiated": The reference does not support the statement. This includes if it contradicts the statement, omits key claims, or addresses unrelated topics.

Your output should be in JSON format with two fields:
{{
  "label": "Substantiated" | "Unsubstantiated",
  "explanation": "A short, clear explanation (1-3 sentences) justifying your label"
}}

Be concise, but ensure your explanation shows your reasoning clearly.

# The citing article

-- Title: {title} 

-- Statement: {statement}
    
# The reference article 

-- Reference Number: {reference_number}

-- Title: {reference_title} 

-- Abstract: {reference_abstract} 

-- Excerpts: \n{reference_excerpts}
"""
    
    return prompt

def create_prompt_suit(df_row, fixed_coverage=False, indications=None):
    title = df_row['Citing Article Title']

    statement = df_row["Corrected Statement"]
    assert statement is not None and statement != '', "Statement cannot be None or empty"

    prompt = f'''   
You are an expert at understanding the capabilities and limitations of large language models, especially in the context of text classification tasks. 
Your task is to determine and explain if a classifier LLM would be capable to successfully verify whether a reference article supports a given citation statement.  

The classifier LLM will be given a citation statement from an article that cites a reference article.
From this same reference article it will receive the additional information, including the title, the abstract of the article and the top 3 most relevant excerpts from the reference article. The relevance of the excerpts was previously determined by another large language model based on the citation statement.
The statement sentence can contain multiple citations and can refer to multiple reference articles, which are all cited in IEEE style.
The classifier LLM is given the number of the reference article that it should check ("Reference Number"). When for example the statement is "X is true [37, 38] and Y is false under certain conditions [39]", and the LLM is given the reference number 37, it should only check the first part of the statement that refers to the reference article 37.

As the classification result, the classifier LLM should decide between the two labels "Substantiated" and "Unsubstantiated". 
Further explanations of the labels are as follows: 
"Substantiated": The reference article fully substantiates the relevant part of the presented citation statement. This means that only based on the information from the reference article, the statement does not contain errors and can be considered correct. 
"Unsubstantiated": The reference article does not substantiate the relevant part of the presented citation statement. This could be because the statement is contradictory to, unrelated to, or simply missing from the reference article. All of these options would indicate that the citation is incorrect based on the cited references.
    
You should evaluate if the following citation statement is suited to be correctly classified by a classifier LLM, using your knowledge on the limitations of large language models. You do not get any of the information on the reference article so that you can focus on the characteristics of the citation itself. Please {'' if indications == 'any' else 'only '}answer “No” if there are{' strong' if indications == 'strong' else ' any' if indications == 'any' else ''} indications that the citation is not suited{'.' if indications == 'any' else ', because your response will be used to perform a selective classification'}.
{'The selective classification coverage should be at around 85 \%' if fixed_coverage else ''}
Format your answer in the following JSON format: 
{{
    "citation suited": “Yes”/”No” (based on your decision),
    "explanation": <reasoning for your decision>
}}
Your explanation should be short and concise. 
    
# The citing article

-- Title: {title} 

-- Statement: {statement}
'''

    return prompt