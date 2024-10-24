from pydantic import BaseModel, conlist
from typing import List, Literal

class MCQOption(BaseModel):
    option_number: int
    text: str

class MCQSet(BaseModel):
    question: str
    chapter_number: int
    options: conlist(MCQOption) # type: ignore
    correct_option_number: int
    chapter_start_timestamp: str
    chapter_end_timestamp: str

class MCQCollection(BaseModel):
    mcq_sets: List[MCQSet]


class SubjectiveQuestion(BaseModel):
    chapter_number: int
    chapter_name: str
    chapter_start_timestamp: str
    chapter_end_timestamp: str
    chapter_question: str
    chapter_answer: str

class SubjectiveCollection(BaseModel):
    subjective_questions: List[SubjectiveQuestion]