import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

import hparams as hp
import audio
import utils
import dataset
import text
import model as M
import waveglow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(
        hp.checkpoint_path, checkpoint_path), map_location=device)['model'])
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).to(device).long()
    src_pos = torch.from_numpy(src_pos).to(device).long()

    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


if __name__ == "__main__":
    # Test
    WaveGlow = utils.get_WaveGlow()
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--say", default="")
    args = parser.parse_args()

    if not os.path.exists("results"):
        os.mkdir("results")
        
    if (args.say==""):
        print("Please use --say argument to specify the text")
        exit()

    print("use griffin-lim and waveglow")
    
    model = get_DNN(args.step)
    
    phn = text.text_to_sequence(args.say, hp.text_cleaners)
    mel, mel_cuda = synthesis(model, phn, args.alpha)

    audio.tools.inv_mel_spec(
        mel, "results/"+str(args.step)+".wav")
    waveglow.inference.inference(
        mel_cuda, WaveGlow,
        "results/"+str(args.step)+"_waveglow.wav")
    
    print("Saved as results/"+str(args.step)+".wav and results/"+str(args.step)+"_waveglow.wav")
    print("Done")
    

