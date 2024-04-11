import whisperx
import gc 
import torch 

device = "gpu" if torch.cuda.is_available() else "cpu"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = None

def init_model():
  global model
  model = whisperx.load_model("large-v2", device, compute_type=compute_type,language="en")

def stt(audio) -> str:
  result = model.transcribe(audio, batch_size=batch_size)
  text = ""
  for t in result["segments"]:
    text +=  t["text"]
  return text
