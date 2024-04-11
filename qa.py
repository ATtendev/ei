
from openai import OpenAI
import os
from dotenv import load_dotenv 
from private_gpt import get_q_a as get_q_a_private
load_dotenv() 

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_q_a_open_ai(p_lang,p_level):
    """
    Retrieves a question and answer based on the provided programming language and level.

    Args:
        p_lang (str): The programming language.
        p_level (str): The programming level.
    Returns:
        q (str): The question.
        a (str): The answer.
    """

    system_content = f"You will be provided with a piece of code, and your task is to explain it concisely and provide a simple example. [Programming Language: {p_lang}, Programming Level: {p_level}]"
    # get user_question from database or somewhere else
    user_question = """ What are lists and tuples? What is the key difference between the two?"""
    user_message = {"role": "user", "content": user_question}
    system_message = {"role": "system", "content": system_content}

    messages = [system_message, user_message]

    response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )    
    print(response.choices[0].message.content)
    return {
        "q": user_question,
        "a": response.choices[0].message.content
    }
 

def get_q_a_private_gpt(p_lang,p_level,p_exp):
    """
    Retrieves a question and answer based on the provided programming language and level.

    Args:
        p_lang (str): The programming language.
        p_level (str): The programming level.
    Returns:
        q (str): The question.
        a (str): The answer.
    """

    return get_q_a_private(p_lang, p_level,p_exp)
 
# using privategpt to Q&A generated
   
if __name__=="__main__":
    # print(get_q_a_open_ai("Python", "Beginner"))
    print(get_q_a_private_gpt("Python", "Beginner",["Golang","Rust"]))