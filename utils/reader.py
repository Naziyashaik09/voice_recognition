import librosa
import numpy as np
from torch.utils import data



def load_audio(audio_path, mode='train', win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257):
   
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    print(audio_path)
   
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    else:
        extended_wav = np.append(wav, wav[::-1])
    
    linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    mag, _ = librosa.magphase(linear)
    freq, freq_time = mag.shape
    assert freq_time >= spec_len, "The length of the non-silent part cannot be less than 1.3s"
    if mode == 'train':
        
        rand_time = np.random.randint(0, freq_time - spec_len)
        spec_mag = mag[:, rand_time:rand_time + spec_len]
    else:
        spec_mag = mag[:, :spec_len]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, :]
    return spec_mag



class CustomDataset(data.Dataset):
    def __init__(self, data_list_path, model='train', spec_len=257):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.model = model
        self.spec_len = spec_len

    def __getitem__(self, idx):
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        spec_mag = load_audio(audio_path, mode=self.model, spec_len=self.spec_len)
        return spec_mag, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)
