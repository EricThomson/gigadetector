"""
Utility to download faster-rcnn trained on fish (fish_frcnn_graph.pb)
into gigadetector/models, and also creates data/ and downloads
three gigaimages that contains with 93 fish in an arena, if not already done.

Code for using api to download from google drive adapted from:
https://stackoverflow.com/a/39225272/1886357


Part of gigadetector repo:
https://github.com/EricThomson/gigadetector
"""
import requests
import os

def download_from_google(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = {'file_id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'file_id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    image_ids = ['1ItmQnjCCxeSJV-7b0umwYEQxA5K8pMyG',
                '1dz9OgOylKfRjukYr0NHJqV_ewVgA6eD9',
                '1MTHUaS2TYBMMa9t3Q1PmCzblQQma07nU']
    model_id = r'1tQirfr0htIpEAF_jlm3sLeMRGisiJbyN'

    base_directory = os.path.expanduser("~") + r"/gigadetector/"
    model_directory = base_directory + "models/"
    data_directory = base_directory + "data/"
    if os.path.isdir(data_directory):
        pass
    else:
        os.mkdir(data_directory)

    # get images
    for ind, image_id in enumerate(image_ids):
        image_path = data_directory + f"giga{ind+1}.png"
        if not os.path.isfile(image_path):
            download_from_google(image_id, image_path)
            print(f"{image_path} downloaded.")
        else:
            print(f"{image_path} already downloaded.")
    # get model
    model_path = model_directory + 'fish_frcnn_graph.pb'
    if not os.path.isfile(model_path):
        download_from_google(model_id, model_path)
        print(f"{model_path} downloaded.")
    else:
        print(f"{model_path} already downloaded.")
