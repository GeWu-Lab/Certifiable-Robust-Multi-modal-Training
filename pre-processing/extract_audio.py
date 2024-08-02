import csv
import os

import numpy as np
import torchaudio
import torch

# Path to save the processed spectrograms
save_path = 'train_spec'

# Path of the wav files
audio_path = 'train_wav/train'

# List of all wav file names
csv_file = 'ks_train_real.txt'

data = []
# Open the CSV file and read each line
with open(csv_file) as f:
    for line in f:
        item = line.split("\n")[0].split(" ")
        name = item[0][:-4]

        # Check if the wav file exists in the specified path
        if os.path.exists(audio_path + '/' + name + '.wav'):
            data.append(name)

# Process each audio file name
for name in data:
    # Load the audio file
    waveform, sr = torchaudio.load(audio_path + '/' + name + '.wav')
    waveform = waveform - waveform.mean()
    norm_mean = -4.503877
    norm_std = 5.141276

    # Compute the Mel spectrogram using torchaudio's Kaldi function
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames

    # Pad or trim the Mel spectrogram to the target length
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = (fbank - norm_mean) / (norm_std * 2)

    print(fbank.shape)
    # Save the processed Mel spectrogram
    np.save(save_path + '/' + name + '.npy', fbank)