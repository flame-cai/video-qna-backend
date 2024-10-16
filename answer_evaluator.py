import openai
import os

from openai import OpenAI
from pydantic import BaseModel


class AnswerEvaluation(BaseModel):
    isCorrect: bool
    explanation: str


def evaluate_answer(question, answer, submission):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful answer validator. You are given a question, its answer and a submission by the user. Your task is to evaluate the submission against the actual answer and if the submission is wrong, provide an explanation on why it is wrong.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\n Answer: {answer}\n Submission: {submission}",
        },
    ]
    client = OpenAI(api_key=os.getenv('API_KEY'))
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0.5,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format=AnswerEvaluation,
        )
        answer_response = completion.choices[0].message
        if answer_response.parsed:
            return answer_response.parsed
        elif answer_response.refusal:
            return answer_response.refusal
    except Exception as e:
        if type(e) == openai.LengthFinishReasonError:
            print("Too many tokens: ", e)
            return {"error": "Try shortening your answer"}
        else:
            print(e)
            return {"error": "Something went wrong."}
