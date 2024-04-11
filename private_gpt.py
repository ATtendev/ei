from pgpt_python.client import PrivateGPTApi
import random
import os
import ast
from dotenv import load_dotenv 

load_dotenv() 

def get_q_a(p_lang,p_level,p_exp):
    #TODO: optimize generate Q&A
    """
    Retrieves a question and answer based on the provided programming language and level.

    Args:
        p_lang (str): The programming language.
        p_level (str): The programming level.
        p_exp (str): The programming early used to determine the experience level.
    Returns:
        q (str): The question.
        a (str): The answer.
    """
    client = PrivateGPTApi(base_url=os.environ.get('PRIVATE_URL'), timeout=360)

    system_content = ""
    if random.random() < 0.5:
        system_content = f"Please provide only 1 principal question along with its answer: [Programming Language: {p_lang}, Programming Level: {p_level}]. Format in the following JSON schema key q for the question and key a for the answer."
    else:
        system_content = f"Please provide only 1 principal question along with its answer, specifying the type of question comparative and relating to programming experience {p_exp}: [Programming Language: {p_lang}, Programming Level: {p_level}]. Format in the following JSON schema key q for the question and key a for the answer."
    
    response = ""   
    for i in client.contextual_completions.prompt_completion_stream(
        prompt=system_content,
        use_context= True,
        include_sources = True
    ):
        response += i.choices[0].delta.content
        
    result = ast.literal_eval(response)
    print("old answer:", result['a'])
    system_content = f"question: {result["q"]}, answer: {result['a']}. Make answer better for programming Level: {p_level}. Response only answer",
    
    final_answer = ""
    for i in client.contextual_completions.prompt_completion_stream(
        prompt=" ".join(system_content),
        use_context= False,
    ):
        final_answer += i.choices[0].delta.content
    
    result['a']=final_answer
    print("new answer:", result['a'])
    return result
 
if __name__ ==  "__main__":
    get_q_a("System design", "Expert", [])