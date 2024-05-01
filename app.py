import queue
from flask import (
    Flask,
    flash,
    request,
    redirect,
    url_for,
    Response,
    send_from_directory,
)
from flask_cors import CORS, cross_origin
from message_announcer import MessageAnnouncer
from utils import predict, hash_file, format_sse
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from eventloopthread import run_coroutine
import json


denoised_images = {}

UPLOAD_FOLDER = "./uploads"
DENOISED_FOLDER = "./denoised/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.add_url_rule("/denoised/<name>", endpoint="download_file", build_only=True)


@app.route("/denoised/<name>")
def download_file(name):
    return send_from_directory(DENOISED_FOLDER, name)


@app.route("/")
def hello_world():
    return "Health check"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/denoise", methods=["POST"])  # type: ignore
@cross_origin()
def denoise():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file and allowed_file(file.filename):
        if file.filename is None:
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        hash = hash_file(filepath)
        if hash in denoised_images:
            return {"id": hash}, 200
        announcer = MessageAnnouncer()
        denoised_images[hash] = announcer
        run_coroutine(
            predict(
                id=hash,
                image_path=filepath,
                save_path=DENOISED_FOLDER + hash + ".jpg",
                announcer=announcer,
            )  # type: ignore
        )
        return {"id": hash}, 200
    return {"error": "File not accepted"}, 400


@app.route("/listen/<string:id>", methods=["GET"])
@cross_origin()
def listen(id):
    if id not in denoised_images:
        return {"error": "No such id found"}, 404
    denoised_image = Path(DENOISED_FOLDER + id + ".jpg")
    if denoised_image.is_file():
        msg = format_sse(
            data=json.dumps(
                {"id": id, "filepath": url_for("download_file", name=id + ".jpg")}
            ),
            event="completed",
        )
        denoised_images[id].announce(msg=msg)

    def stream():
        messages = denoised_images[id].listen()  # returns a queue.Queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
