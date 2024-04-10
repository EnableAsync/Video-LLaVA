import shutil
import subprocess

import torch
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
from decord import VideoReader, cpu
from transformers import TextStreamer

from videollava.constants import DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle, Conversation
from videollava.serve.gradio_utils import Chat, tos_markdown, learn_more_markdown, title_markdown, block_css

import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


conv_mode = "llava_v1"
model_path = 'checkpoints/videollava-7b-lora'
cache_dir = './cache_dir'
device = 'cuda:0'
load_8bit = False
load_4bit = False
dtype = torch.float16
handler = Chat(model_path, conv_mode=conv_mode, load_8bit=load_8bit, load_4bit=load_8bit, device=device, cache_dir=cache_dir)

data_root = "dataset/llava_image_tune/"

def parse_my_json(file_path):
    with open(file_path, "r") as f:
        json_data = f.read()
    labels = json.loads(json_data)
    return labels

def generate(image1, text_en_in):
    state = gr.State()
    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        images_tensor = []

    image_processor = handler.image_processor
    if os.path.exists(data_root + image1):
        tensor = image_processor.preprocess(data_root + image1, return_tensors='pt')['pixel_values'][0]
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor.append(tensor)
        text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + text_en_in
    text_en_out, state_ = handler.generate(images_tensor, text_en_in, first_run=True, state=state_)
    return text_en_out

def get_acoustic_category_index(label):
    if 'bending' in label:
        return 0
    elif 'falling quickly' in label:
        return 1
    elif 'jumping' in label:
        return 2
    elif 'falling slowly' in label:
        return 3
    elif 'squatting' in label:
        return 4
    elif 'standing still' in label:
        return 5
    elif 'nobody' in label:
        return 6
    elif 'picking' in label:
        return 7
    elif 'motion of a person standing' in label:
        return 8
    elif 'walking':
        return 9

def get_wifi_category_index(label):
    if 'L-shaped' in label:
        return 0
    elif 'O-shaped' in label:
        return 1
    elif 'V-shaped' in label:
        return 2
    elif 'S-shaped' in label:
        return 3
    elif 'W-shaped' in label:
        return 4
    elif 'Z-shaped' in label:
        return 5

def eval_acoustic():
    i = 0
    data = parse_my_json("dataset/metadata/acoustic_test.json")

    y_true = []
    y_pred = []

    for obj in data:
        image = obj["image"]
        label = obj["conversations"][1]['value']
        ret = generate(image, 'Briefly describe this image.')

        y_true.append(get_acoustic_category_index(label))
        y_pred.append(get_acoustic_category_index(ret))

        if ret == label:
            i += 1
        else:
            print(obj)
            print(ret)

    cm = confusion_matrix(np.array(y_true), np.array(y_pred))
    ticklabels = ['bending', 'falling quickly', 'jumping', 'falling slowly', 'squatting', 'standing still', 'nobody', 'picking', 'standing up', 'walking']
    # 可视化混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ticklabels, yticklabels=ticklabels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)  # 旋转类别标签
    plt.yticks(rotation=45)  # 旋转类别标签
    plt.show()
    plt.savefig('acoustic_cm.png')
    print(i / len(data))

def eval_wifi():
    i = 0
    data = parse_my_json("dataset/metadata/wifi_test.json")

    y_true = []
    y_pred = []

    for obj in data:
        image = obj["image"]
        label = obj["conversations"][1]['value']
        ret = generate(image, 'Briefly describe this image.')

        y_true.append(get_wifi_category_index(label))
        y_pred.append(get_wifi_category_index(ret))

        if ret == label:
            i += 1
        else:
            print(obj)
            print(ret)
    
    cm = confusion_matrix(np.array(y_true), np.array(y_pred))
    ticklabels = ['L', 'O', 'V', 'S', 'W', 'Z']
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ticklabels, yticklabels=ticklabels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('wifi_cm.png')
    print(i / len(data))




if __name__ == "__main__":
    # 0.67
    eval_acoustic()

    # 0.78
    # eval_wifi()
