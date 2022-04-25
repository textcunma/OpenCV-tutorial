"""
flaskを用いてカメラ映像に画像処理を行う
fkask x OpenCV

参考 https://qiita.com/Gyutan/items/1f81afacc7cac0b07526
"""

from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")  # tmplates/index.htmlを読み込む


def generator(camera):
    """
    input : VideoCamera class
    output : frame
    """
    while True:
        frame = camera.get_frame()  # カメラ映像から画像を取得a

        # 逐次出力
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        generator(VideoCamera()), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(port=8000, debug=True)
