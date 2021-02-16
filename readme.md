# Gigadetector
Object detection pipeline when your images are too big to fit into your GPU.

<describe basic process here>

## Installation
### Clone repo / create  virtual environment
For now (no installer): go to the directory that you want to contain the `gigadetector` repo:    

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

If you get `False` try entering `nvidia-smi` at the OS command line: if you do not get a reasonable response from nvidia, you probably need to troubleshoot the GPU-Driver-Cuda triad -- an ML rite of passage sent from the depths of hell.

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

Update python path:

    cd ~/models/research/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Now you should have access to all the features of the object detection api! Test it with the following:

Test your install:

      cd ~/models/research
      python object_detection/builders/model_builder_tf1_test.py

If you see a bunch of lines indicating tests were Run (maybe with one or two skipped), you are good to go.

### Test the workflow on a sample small
Test the model on a single test image from the command line.

    conda activate gigadetector
    cd ~/gigadetector/gigadetector/
    python predict_one.py --model /home/naumann/gigadetector/models/fish_frcnn_graph.pb --labels /home/naumann/gigadetector/models/fish_classes.pbtxt --image /home/naumann/gigadetector/test_data/fish.png --num-classes 1

You will see some warnings, but you should see an image pop up with two fish outlined with the confidence measure (in this case it will be rounded up to 1.00 for both fish).

## Test on a single gigaimage
To do: add description and docs here for this (will cycle through image).

    cd ~/gigadetector
    conda activate gigadetector/gigadetector
    python gigatest.py

Then predict_loop_script.py (with bb_draw.py)

## Run in a folder of images
Then predict_folder.py (wth bb_analysis_folder.py)
Then gigaviewer.py

## To do
- push online to a private repo
- let mark and colin know.

### Acknowledgments
Code developed as part of the multi-camera array microscope project in the Computational Optics Lab (http://horstmeyer.pratt.duke.edu/) and the Naumann Lab (https://www.naumannlab.org/) at Duke University.
