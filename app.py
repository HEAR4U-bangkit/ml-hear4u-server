import io
from fastapi import FastAPI
import tensorflow as tf
import socketio
from pydub import AudioSegment

# Initialize FastAPI and Socket.IO for real-time sound classification usable in Android Kotlin via microphone
app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

# Load the model
loaded_model = tf.saved_model.load("model")

# Preprocess audio to 16k
TARGET_SAMPLE_RATE = 16000

my_classes = ['crying_baby', 'door_knock', 'glass_breaking', 'siren', 'car_horn', 'train', 'door_bells', 'cat', 'dog', 'gun_shot']

def preprocess(audio_data):
    audio = tf.audio.decode_wav(audio_data, desired_channels=1, desired_samples=TARGET_SAMPLE_RATE)
    audio = tf.squeeze(audio.audio, axis=-1)
    return audio

def predict(audio):
    result = loaded_model(audio)
    top_class = tf.math.argmax(result)
    probably = tf.nn.softmax(result, axis=-1)
    your_top_score = probably[top_class]

    infered_class = my_classes[top_class]

    return infered_class, int(your_top_score*100)

# Convert audio bytes to WAV format and process
def convert_and_preprocess(data):
    audio = AudioSegment.from_file(io.BytesIO(data), format="raw", frame_rate=44100, channels=1, sample_width=2)
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
    audio_data = io.BytesIO()
    audio.export(audio_data, format="wav")
    audio_data.seek(0)
    return preprocess(audio_data.read())

@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")
    await sio.emit('response', {'message': 'Connected to server'})

@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")

@sio.event
async def predict_audio(sid, data):
    try:
        audio = convert_and_preprocess(data)
        result, confidence = predict(audio)

        if confidence < 70:
            await sio.emit('response', {'label': "not_detect", 'confidence': confidence}, to=sid)
        else:
            await sio.emit('response', {'label': result, 'confidence': confidence}, to=sid)

    except Exception as e:
        await sio.emit('response', {'error': str(e)}, to=sid)

@app.get('/')
def hello_world():
  return {"test": "hello"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
