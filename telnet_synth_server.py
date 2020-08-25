import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os
import sys
import socket

import hparams as hp
import audio
import utils
import dataset
import text
import model as M
import waveglow

# TCP Server configuration
ListenPort = 8021
Host = ''

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

# Telnet-style readline
# Supports backspace
def ReadLine(sock):
    str = ""
    
    while True:
        chunk = sock.recv(1024)
        if chunk == b'':
            return False
        if chunk.decode() == '\n':
            return str
        if chunk.decode() == '\r\n':
            return str
        
        # Backspace handling
        if chunk.decode() == '\b':
            str = str[:-1]
            sock.send(" \b".encode())
            continue
        
        str = str + chunk.decode()
        
def WriteLine(sock, line):
    line = line + "\r\n"
    sock.send(line.encode())

if __name__ == "__main__":
    # TCP Server Init
    MainSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    MainSocket.bind((Host, ListenPort))
    MainSocket.listen()
    
    # Network init
    WaveGlow = utils.get_WaveGlow()
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists("results"):
        os.mkdir("results")
    
    model = get_DNN(args.step)
    
    print("Waiting for connections...")
    
    while True:
        Client, Addr = MainSocket.accept()
        print("New connection!")
        
        while Client:
            Client.send(">".encode())
            
            Reply = ReadLine(Client)
            
            if Reply:
                Reply = Reply.strip()
                
                phn = text.text_to_sequence(Reply, hp.text_cleaners)
                mel, mel_cuda = synthesis(model, phn, args.alpha)

                waveglow.inference.inference(
                    mel_cuda, WaveGlow,
                    "results/"+str(args.step)+"_waveglow.wav")

                WriteLine(Client, "Saved as results/"+str(args.step)+"_waveglow.wav")
                WriteLine(Client, "Done")
        

