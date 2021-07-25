import torch
import fairseq
import librosa
import transformers

cp_path = '/path/to/wav2vec.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])

speech , rate = librosa.load("Stairway_to_heaven",sr=16000)

input_values = tokenizer(speech,return_tensors='pt').input_values()



