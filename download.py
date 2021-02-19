"""
Utility to download three sample images and the frozen
faster-rcnn model trained on fish (fish_frcnn_graph.pb)

Part of gigadetector repo:
https://github.com/EricThomson/gigadetector
"""
import gdown
import os

base_directory = os.path.expanduser("~") + r"/gigadetector/"
model_directory = base_directory + "models/"
data_directory = base_directory + "data/"
if os.path.isdir(data_directory):
    pass
else:
    os.mkdir(data_directory)

image_ids = ['1ItmQnjCCxeSJV-7b0umwYEQxA5K8pMyG',
            '1dz9OgOylKfRjukYr0NHJqV_ewVgA6eD9',
            '1MTHUaS2TYBMMa9t3Q1PmCzblQQma07nU']
model_id = r'1tQirfr0htIpEAF_jlm3sLeMRGisiJbyN'

# download images
base_url = r'https://drive.google.com/uc?id='
for ind, image_id in enumerate(image_ids):
    image_path = data_directory + f"giga{ind+1}.png"
    if not os.path.isfile(image_path):
        url = base_url + image_id
        gdown.download(url, image_path, quiet=False)
    else:
        print(f"{image_path} already downloaded.")

# download model
model_path = model_directory + 'fish_frcnn_graph.pb'
if not os.path.isfile(model_path):
    url = base_url + model_id
    gdown.download(url, model_path, quiet=False)
else:
    print(f"{model_path} already downloaded.")
