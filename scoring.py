from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

EVALUATION_PROMPT_TEMPLATE = """
You will be provided with one answer to a programming-related question. Your task is to rate the answer based on its programming proficiency level using a specified metric. Please ensure that you thoroughly comprehend these instructions. Keep this document accessible while reviewing and refer to it as necessary for guidance.


Evaluation Criteria:
- Programming Level: {level}
- Programming Language: {language}
{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Answer:

{answer}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Metric 1: Programming Proficiency

PROFICIENCY_SCORE_CRITERIA = """
Programming Proficiency (1-5) - assessment of the depth of programming knowledge and expertise demonstrated in the answer.
1: Novice. The answer lacks fundamental understanding and demonstrates minimal programming knowledge.
2: Beginner. The answer shows some basic understanding but lacks depth and may contain significant errors.
3: Intermediate. The answer demonstrates a moderate level of programming knowledge and understanding.
4: Advanced. The answer exhibits a high level of programming proficiency with clear understanding and minimal errors.
5: Expert. The answer showcases exceptional programming expertise, depth of knowledge, and minimal to no errors.
"""

PROFICIENCY_SCORE_STEPS = """
1. Carefully read the provided answer to the programming-related question.
2. Evaluate the depth of programming knowledge demonstrated in the answer.
3. Consider the accuracy, clarity, and sophistication of the programming concepts presented.
4. Assign a programming proficiency score from 1 to 5 based on the given criteria.
"""

# Metric 2: Clarity

CLARITY_SCORE_CRITERIA = """
Clarity (1-5) - assessment of how clearly the programming concepts are explained in the answer.
1: Unclear. The answer is confusing, poorly organized, and difficult to understand.
2: Somewhat Clear. The answer provides some clarity but may contain ambiguities or lack coherence.
3: Clear. The answer presents programming concepts in a straightforward manner, but may occasionally be unclear or verbose.
4: Very Clear. The answer is well-structured, concise, and easy to understand with minimal ambiguity.
5: Extremely Clear. The answer is exceptionally well-written, precise, and leaves no room for misunderstanding.
"""

CLARITY_SCORE_STEPS = """
1. Review the provided answer to the programming-related question.
2. Assess the clarity of the explanations and examples provided in the answer.
3. Consider the organization, coherence, and conciseness of the answer.
4. Assign a clarity score from 1 to 5 based on the given criteria.
"""

# Metric 3: Completeness

COMPLETENESS_SCORE_CRITERIA = """
Completeness (1-5) - evaluation of how comprehensively the answer addresses all aspects of the programming question.
1: Incomplete. The answer lacks crucial information and fails to address significant aspects of the question.
2: Partially Complete. The answer covers some but not all aspects of the question, leaving important details or concepts unaddressed.
3: Moderately Complete. The answer addresses most aspects of the question but may lack depth or overlook minor details.
4: Mostly Complete. The answer provides a thorough coverage of the question with few omissions or oversights.
5: Fully Complete. The answer is exhaustive, addressing all aspects of the question in depth with precision and clarity.
"""

COMPLETENESS_SCORE_STEPS = """
1. Examine the provided answer in relation to the programming question.
2. Determine if all essential aspects of the question are addressed in the answer.
3. Consider the depth of coverage and whether any critical details are missing.
4. Assign a completeness score from 1 to 5 based on the given criteria.
"""

# Metric 4: Accuracy

ACCURACY_SCORE_CRITERIA = """
Accuracy (1-5) - assessment of the correctness and precision of the programming concepts presented in the answer.
1: Inaccurate. The answer contains numerous factual errors and demonstrates a lack of understanding.
2: Mostly Inaccurate. The answer is primarily incorrect or misleading, with only a few accurate statements.
3: Moderately Accurate. The answer contains some accurate information but also includes significant errors or misconceptions.
4: Mostly Accurate. The answer is mostly correct, with minor errors or inaccuracies.
5: Completely Accurate. The answer is entirely correct, demonstrating a precise understanding of the programming concepts.
"""

ACCURACY_SCORE_STEPS = """
1. Evaluate the accuracy of the programming concepts and explanations provided in the answer.
2. Verify the correctness of any code examples or technical details presented.
3. Consider any potential misconceptions or inaccuracies in the answer.
4. Assign an accuracy score from 1 to 5 based on the given criteria.
"""

evaluation_metrics = {
    "Proficiency": (PROFICIENCY_SCORE_CRITERIA, PROFICIENCY_SCORE_STEPS),
    "Clarity": (CLARITY_SCORE_CRITERIA, CLARITY_SCORE_STEPS),
    "Completeness": (COMPLETENESS_SCORE_CRITERIA, COMPLETENESS_SCORE_STEPS),
    "Accuracy": (ACCURACY_SCORE_CRITERIA, ACCURACY_SCORE_STEPS),
}


def get_geval_score(
    criteria: str, steps: str, document: str, answer: str, metric_name: str,language: str, level: str
):

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        answer=answer,
        level= level,
        language=language,
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content



def get_score(excerpt: str, answer: str,language: str, level: str):
    """
    return {
        "Proficiency": 4,
        "Clarity": 5,
        "Completeness": 5,
        "Accuracy": 5
    }
    """
    data = {}
    for eval_type, (criteria, steps) in evaluation_metrics.items():
            result = get_geval_score(criteria, steps, excerpt, answer, eval_type,language,level)
            data[eval_type] = int(result.strip())
    return data


# Using ChatGPT or any other tool can help candidates improve their answers.
IMPROVE_PROMPT_TEMPLATE = """
I want you to act as an AI assistant that can provide helpful comments to explain answers.
Interviewer : {question}
Interviewee : {answer}
Interviewee Level : {level}
"""


def improve_answer(question: str, answer: str, level: str):
    prompt = IMPROVE_PROMPT_TEMPLATE.format(
    question=question,
    answer=answer,
    level=level,
    )
    # return prompt
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content

if __name__== "__main__":
    question ="""What are lists and tuples? What is the key difference between the two?"""
    excerpt = """Lists and Tuples are both sequence data types that can store a collection of objects in Python. The objects stored in both sequences can have different data types. Lists are represented with square brackets ['sara', 6, 0.19], while tuples are represented with parantheses ('ansh', 5, 0.97).
But what is the real difference between the two? The key difference between the two is that while lists are mutable, tuples on the other hand are immutable objects. This means that lists can be modified, appended or sliced on the go but tuples remain constant and cannot be modified in any manner."""
    # eval_answer = """In Python, lists and tuples are both used to store multiple items in a single variable. The key difference between them is that lists are mutable (can be changed), while tuples are immutable (cannot be changed)."""
    eval_answer = """GPT-3.5 Turbo models can understand and generate natural language or code and have been optimized for chat using the Chat Completions API but work well for non-chat tasks as well."""
    # print(improve_answer(question, eval_answer, "Beginner"))
    print(get_score(excerpt, eval_answer,"Python", "Expert"))
    