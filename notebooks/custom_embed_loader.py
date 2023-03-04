import torch
import os


def load(fpath='embeds'):
    custom_embeds = {}

    files = [fpath + '/' + f for f in os.listdir('embeds') if f.endswith(".bin")]
    for file in files:
        loaded_learned_embeds = torch.load(file, map_location="cpu")
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token].detach().numpy()
        custom_embeds[trained_token] = embeds

    return custom_embeds
