import torch
import fairseq
import librosa



cp_path = r'C:\Users\ashok kumar\Documents\fairseq/wav2vec_small.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])

speech , rate = librosa.load("Stairway_to_Heaven",sr=16000)
processor = Wav2Vec2Processor.from_pretrained("wav2vec_small.pt")


z = model.feature_extractor(speech)
c = model.feature_aggregator(z)



