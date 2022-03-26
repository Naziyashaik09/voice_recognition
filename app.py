
from cProfile import label
from curses.ascii import NUL
import gradio as gr

import argparse
import functools
import distutils.util

import numpy as np
import torch
from infer_contrast import run
#import utils
from utils.reader import load_audio
from utils.utility import add_arguments, print_arguments

STYLE = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" integrity="sha256-YvdLHPgkqJ8DVUxjjnGVlMMJtNimJ6dYkowFFvp4kKs=" crossorigin="anonymous">
"""
OUTPUT_OK = (
    STYLE
    + """
    <div class="container">
        <div class="row"><h1 style="text-align: center">Speaker1 and Speaker2</h1></div>
        <div class="row"><h1 class="display-1 text-success" style="text-align: center">the same person</h1></div>
        <div class="row"><h1 style="text-align: center">Similarity is:</h1></div>
        <div class="row"><h1 class="display-1 text-success" style="text-align: center">{:.1f}%</h1></div>
        <div class="row"><small style="text-align: center">(A similarity of more than 70% can be considered as the same person)</small><div class="row">
    </div>
"""
)
OUTPUT_FAIL = (
    STYLE
    + """
    <div class="container">
        <div class="row"><h1 style="text-align: center">Speaker1 and Speaker2</h1></div>
        <div class="row"><h1 class="display-1 text-danger" style="text-align: center">not the same person</h1></div>
        <div class="row"><h1 style="text-align: center">Similarity is:</h1></div>
        <div class="row"><h1 class="text-danger" style="text-align: center">{:.1f}%</h1></div>
        <div class="row"><small style="text-align: center">(A similarity of more than 70% can be considered as the same person)</small><div class="row">
    </div>
"""
)

THRESHOLD = 0.70
def voiceRecognition(audio1,audio2,recording1,recording2):
    #if audio1 and audio2 is None:
    #score=run(recording1,recording2)
    print (type(recording1))
    print(type(audio1))
    #if recording1 and recording2 is None:
    score=0
 #if recording1 and recording2 is None:
    if recording1 and recording2 != None:
     print("reocrding is not none")
     score = run(recording1,recording2)
     if audio1 and audio2 != None:
      print("audio is not none")
     score = run(audio1,audio2)
    if score >= THRESHOLD:
        output = OUTPUT_OK.format(score * 100)
    else:
        output = OUTPUT_FAIL.format(score * 100)
    return output




title = "Voice Recognition"
description = "Choose one of the option ie., either upload(speaker1 and speaker2) OR record(speaker@1 and speaker@2),Don't choose multiple options"
            

inputs = [gr.inputs.Audio(source='upload',type="filepath",optional=True,label="Speaker1"),
          gr.inputs.Audio(source="upload",type="filepath",optional=True,label="Speaker2"),
          gr.inputs.Audio(source="microphone", type="filepath", optional=True, label="Speaker@1"),
          gr.inputs.Audio(source="microphone", type="filepath", optional=True, label="Speaker@2"),
]
output = [gr.outputs.HTML(label="")
]


examples = [
    ["samples/Li Yunlong 1.wav", "samples/Li Yunlong 2.wav"],
    ["samples/Jay Chou 1.wav", "samples/Jay Chou 2.wav"],
    ["samples/Ma Baoguo 1.wav", "samples/Li Yunlong 2.wav"],
    ["samples/Jay Chou 1.wav", "samples/SpongeBob SquarePants 1.wav"],
    ["samples/Pai Daxing.wav", "samples/Ma Baoguo 2.wav"],
    ["samples/Wn Mengda.wav", "samples/Stephen Chow.wav"]]

interface = gr.Interface(
    fn=voiceRecognition,
    inputs=inputs,
    
    outputs=output,
    title=title,
    description=description,
    examples=examples,
    theme='dark',
    enable_queue=True)
interface.launch(debug=True,share=True)