import uuid
import json
import threading

from flask import Flask, session, request
from flask_cors import CORS
from redis import Redis

from video_qna_generator import generate_video_qna
from answer_evaluator import evaluate_answer

app = Flask(__name__)
redis = Redis(host='localhost', port=6379, db=0)

app.config.from_mapping(
    SECRET_KEY = 'secret',
)

CORS(app)

@app.route("/")
def hello():
    return "Video QnA Generator"

def qna_generator_task(task_id, url):
    try:
        chapters = generate_video_qna(url)
        redis.set(task_id, json.dumps({"status": "completed", "data": chapters}))
    except Exception as e:
        redis.set(task_id, json.dumps({"status": "failed", "error": str(e)}))

@app.route("/generate-video", methods=["POST"])
def get_video_qna():
    task_id = str(uuid.uuid4())
    redis.set(task_id, json.dumps({"status": "processing"})) 
    url = request.json.get("url")
    if url:
        thread = threading.Thread(target=qna_generator_task, args=(task_id, url))
        thread.start()
        return {"taskId": task_id}, 202

    else:
        return {
            "status": "error",
            "message": "No URL provided in the query parameters.",
        }, 400

@app.route('/generate-video/<task_id>', methods=['GET'])
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
    
@app.route("/evaluate-answer", methods=["POST"])
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
        print(answer_evaluation.model_dump())
        return response, 200
    else:
        return {
            "status": "error",
            "message": answer_evaluation["error"]
        }, 500
