from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model

app = Flask(__name__)

def predict(audio):
    model=load_model('model.hdf5')
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            """
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)"""
            samples, sample_rate = librosa.load(file.filename, sr = 16000)
            samples = librosa.resample(samples, sample_rate, 8000)
            #ipd.Audio(samples,rate=8000)     
            predict(samples)

    return render_template('index.html', transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)