from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import tensorflow_io as tfio
from pydub import AudioSegment

app = FastAPI()

# Load the model
loaded_model = tf.saved_model.load("model")

# Preprocess audio change to 16k
TARGET_SAMPLE_RATE = 16000

my_classes = ['crying_baby', 'door_knock', 'glass_breaking', 'siren', 'car_horn', 'train', 'door_bells', 'cat', 'dog',  'gun_shot']

def preprocess(filename):
  file_contents = tf.io.read_file(filename)
  wav, sample_rate = tf.audio.decode_wav(
    file_contents,
    desired_channels=1)

  wav = tf.squeeze(wav, axis=-1)
  sample_rate = tf.cast(sample_rate, dtype=tf.int64)
  wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=TARGET_SAMPLE_RATE)
  
  return wav

def predict(audio):
  result = loaded_model(audio)
  top_class = tf.math.argmax(result)
  probably = tf.nn.softmax(result, axis=-1)
  your_top_score = probably[top_class]

  infered_class = my_classes[top_class]

  return infered_class, int(your_top_score*100)

# === Convert
def convert_to_wav(file_path, target_path):
  audio = AudioSegment.from_file(file_path)
  audio = audio.set_channels(1)
  audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
  audio.export(target_path, format="wav")

@app.post("/predict")
def predict_sound(file: UploadFile = File(...)):
  try:
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb+") as file_object:
      file_object.write(file.file.read())

    wav_file_location = f"/tmp/converted_{file.filename}"
    convert_to_wav(file_location, wav_file_location)

    audio = preprocess(wav_file_location)

    # audio = preprocess(file_location)

    result, confident = predict(audio)

    return {"label": result, "confident": confident}

  except Exception as e:
    return {"error": str(e)}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)