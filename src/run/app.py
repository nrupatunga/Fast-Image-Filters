"""
File: app.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Demo app for Fast Image filters
"""
from imutils import paths
import gradio as gr
import numpy as np
import torch

from core.trainers.filter_trainer import LitModel

smooth_ckpt = './ckpt/L0-smoothing/_ckpt_epoch_79.ckpt'
style_ckpt = './ckpt/Photographic-Style/epoch=138.ckpt'
pencil_ckpt = './ckpt/Pencil/epoch=138.ckpt'
tone_ckpt = './ckpt/Multiscale-Tone/epoch=178.ckpt'
gpu = True
models = {}


def load_model(ckpt_path):
    """Model load for testing"""
    if gpu:
        model = LitModel.load_from_checkpoint(ckpt_path).cuda()
    else:
        model = LitModel.load_from_checkpoint(ckpt_path,
                                              map_location=torch.device('cpu'))
    model.eval()
    model.freeze()

    return model


def get_output(model, img):
    data = img / 255.
    data = np.transpose(data, [2, 0, 1])
    data = torch.tensor(data).unsqueeze(0)
    data = data.float()

    if gpu:
        output = model(data.cuda()).squeeze().cpu().numpy()
    else:
        output = model(data).squeeze().numpy()

    output = np.transpose(output, [1, 2, 0])
    output = np.clip(output, 0, 1)
    output = output * 255
    output = output.astype(np.uint8)

    return output


def apply_filter(image, mode):
    """apply filter
    """
    return get_output(models[mode], image)


if __name__ == "__main__":
    filter_keys = ['L0-Smoothing', 'Photographic-Style',
                   'Multiscale-Tone', 'Pencil']
    models[filter_keys[0]] = load_model(smooth_ckpt)
    models[filter_keys[1]] = load_model(style_ckpt)
    models[filter_keys[2]] = load_model(tone_ckpt)
    models[filter_keys[3]] = load_model(pencil_ckpt)

    title = 'Fast Image Filters using CNN'
    description = 'Implementation of image filters using CNN'

    inputs = [gr.inputs.Image(), gr.inputs.Radio(filter_keys)]
    outputs = gr.outputs.Image()

    images_dir = './images'
    images_path = paths.list_images(images_dir)
    sample_images = []
    for img_path in images_path:
        sample_images.append([img_path])

    gr.Interface(fn=apply_filter,
                 inputs=inputs,
                 outputs=outputs,
                 title=title,
                 allow_flagging=False,
                 examples=sample_images,
                 description=description).launch()
