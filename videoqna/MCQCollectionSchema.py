from pydantic import BaseModel, conlist
from typing import List, Literal

class MCQOption(BaseModel):
    option_number: int
    text: str

class MCQSet(BaseModel):
    question: str
    options: conlist(MCQOption, min_length=4, max_length=4)
    correct_option_number: int
    chapter_starting_timestamp: str
    chapter_ending_timestamp: str

class MCQCollection(BaseModel):
    mcq_sets: List[MCQSet]