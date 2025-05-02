from flask import Flask, request
from flask_sock import Sock
import json
import requests  # For sending audio chunks to AssemblyAI / Vosk
from kafka import kafka_producer

app = Flask(__name__)
sock = Sock(app)

@sock.route('/media')
def media_stream(ws):
    """
    Receives audio chunks from Twilio Media Streams,
    forwards to AssemblyAI for real-time STT.
    """
    while True:
        msg = ws.receive()
        data = json.loads(msg)
        audio_chunk = data.get('media', {}).get('payload')
        # send to STT service
        resp = requests.post(
            "https://api.assemblyai.com/v2/stream",
            headers={"authorization": "API_KEY"},
            data=audio_chunk
        )
        text = resp.json().get("text", "")
        # push transcript to Kafka for downstream processing
        kafka_producer.send('call_transcripts', text.encode())
    return ""
