import uuid
import json
import threading

from flask import Blueprint, request
from redis import Redis

from videoqna.video_qna_generator import generate_video_qna
from videoqna.answer_evaluator import evaluate_answer

bp = Blueprint('main', __name__)
redis = Redis(host='localhost', port=6379, db=0)

@bp.route("/")
def hello():
    return "Video QnA Generator"

def qna_generator_task(task_id, url, question_format):
    try:
        chapters, duration = generate_video_qna(url, question_format)
        redis.set(task_id, json.dumps({"status": "completed", "data": {"chapters" : chapters, "duration" : duration}}))
    except Exception as e:
        redis.set(task_id, json.dumps({"status": "failed", "error": str(e)}))

@bp.route("/generate-video", methods=["POST"])
def get_video_qna():
    task_id = str(uuid.uuid4())
    redis.set(task_id, json.dumps({"status": "processing"})) 
    url = request.json.get("url")
    question_format = request.json.get("question_format")
    if url:
        thread = threading.Thread(target=qna_generator_task, args=(task_id, url, question_format))
        thread.start()
        return {"taskId": task_id}, 202

    else:
        return {
            "status": "error",
            "message": "No URL provided in the query parameters.",
        }, 400

@bp.route('/generate-video/<task_id>', methods=['GET'])
def task_status(task_id):
    task_data = redis.get(task_id)
    if task_data:
        print(task_data)
        task_data = json.loads(task_data.decode('utf-8'))
        if task_data["status"] == "failed":
            return {
                "taskId": task_id,
                "status": "failed",
                "error": task_data.get("error")
            }, 500
        return {"taskId": task_id, "status": task_data["status"], "data": task_data.get("data")}, 200
    else:
        return {"error": "Task not found"}, 404
    
@bp.route("/evaluate-answer", methods=["POST"])
def do_answer_validation():
    data = request.json
    question = data.get("  ")
    answer = data.get("answer")
    submission = data.get("submission")
    answer_evaluation = evaluate_answer(question, answer, submission)
    if "error" not in answer_evaluation:
        response =  {
            "status": "success",
            "message": f"answer '{submission}' received and processed.",
            "data": answer_evaluation.model_dump(),
        }
        print(response)
        return response, 200
    else:
        return {
            "status": "error",
            "message": answer_evaluation["error"]
        }, 500
