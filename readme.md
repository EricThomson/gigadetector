# Gigadetector
Object detection pipeline when your images are too big to fit into your GPU.

Run and tested on Linux.

## Installation
### Clone repo / create  virtual environment
For now (no installer): go to your home directory:    

    cd ~
    git clone https://github.com/EricThomson/gigadetector.git
    cd gigadetector
    conda env create -f environment.yml
    conda activate gigadetector
    pip uninstall opencv-python

The above command will create the gigadetector virtual environment and activate it (and uninstall a version of opencv we do not need).  Note that creating this virtual environment can take several minutes.

### Test your installation of tensorflow
From the command line, go into python, import tensorflow, and see if you have a gpu available:

    python
    import tensorflow as tf
    tf.test.is_gpu_available()
    exit()

If you have a working version of `tensorflow-gpu`, and a working driver and GPU installed, You should get a bunch of stuff that looks like gibberish, with the final line `True`.

If you get `False` try entering `nvidia-smi` at the OS command line: if you do *not* get a reasonable response, you probably need to troubleshoot the GPU-Driver-Cuda triad -- an ML rite of passage sent from the depths of hell. Good luck.

### Install the tensorflow object detection api
This API contains many useful utilities used by gigadetector that are provided by Google. For more on the API see:
 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md.

 The following steps will install the API, and compile the protobuf libraries used by the api (protocol buffers are serializers used to pull data into the models).

    cd ~
    git clone https://github.com/tensorflow/models
    cd ~/models/research/
    protoc object_detection/protos/*.proto --python_out=.

Now install the object detection api:

    cp object_detection/packages/tf1/setup.py .
    python -m pip install --use-feature=2020-resolver .

Update python path so you can access the utilities in your python scripts:

    cd ~/models/research/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Test it with the following:

      cd ~/models/research
      python object_detection/builders/model_builder_tf1_test.py

If you see a bunch of lines indicating tests were Run (maybe with one or two skipped), you are good to go!

### Download the model and test images
Run the script to download the frozen model (`fish_frcnn_graph.pb`) into `models/` and to create the local `data/` folder and download three large images into this folder. In your gigadetector environment:

    cd ~/gigadetector
    python download_demo.py

You should see some indications that you are downloading images and the frozen fish faster rcnn network (`fish_frcnn.pb`).

### Test on small image
Test the model on a small test image that contains two fish.

    conda activate gigadetector
    cd ~/gigadetector/gigadetector/
    python predict_one.py

You will see some warnings, but you should see an image pop up with two fish outlined with the confidence measure (in this case it will be rounded up to 1.00 for both fish).

Note in what follows the first two lines from above (activating the environment and cd'ing will be assumed).

## Test on a single gigaimage
To do: add description and docs here for this (will cycle through image).

    python gigatest.py

You will see the algorithm run through an entire single gigaimage with a sliding window (it may run relatively quickly)  [*describe more what this does and what/where it saves*].

Add something to draw and show *all* the bboxes.

Describe and mention bb_draw.py anw what/.where it saves.  


## Run in a folder of images
Here';s where we get to something that is more like what you will actually do.
Then predict_folder.py (wth bb_analysis_folder.py)
Then gigaviewer.py

## To do
- refactor/improve documentation (minimal for now)
    - better filenames
    - change to relative paths or dummy paths in readme
- push online to a private repo
- test on my linux machine
- let folks know


### Acknowledgments
Code developed as part of the multi-camera array microscope project in the Computational Optics Lab (http://horstmeyer.pratt.duke.edu/) and the Naumann Lab (https://www.naumannlab.org/) at Duke University.
