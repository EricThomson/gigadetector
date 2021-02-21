# Gigadetector
Object detection pipeline when your images are too big to fit into your GPU. This readme walks you through installation and running simple examples using a frozen model for fish detection. The best way to use it is to adapt the examples for use with your own data. You don't have to use the fish model -- you should be able to download your own frozen model and use this pipeline for any objects.

## Install and test the workflow
When first getting this started, I recommend you go through all of the following steps and tests in order. It was created/tested only on Linux.

### Create  virtual environment
Clone the repository and create a virtual environment (we'll call the virtual environment `gigadetector` but feel free to rename):

    cd ~
    git clone https://github.com/EricThomson/gigadetector.git
    cd gigadetector
    conda env create -f environment.yml
    conda activate gigadetector
    pip uninstall opencv-python

The above command will create the gigadetector virtual environment and activate it (and uninstall a superfluous version of opencv that one of the dependencies installs).  Note creating this virtual environment typically takes several minutes.

### Test tensorflow installation
From the command line, go into python, import tensorflow, and make sure it sees your gpu:

    python
    import tensorflow as tf
    tf.test.is_gpu_available()

If you have a working version of `tensorflow-gpu` that sees your gpu, you should get a bunch of gibberish, and the final line `True`.

If you get `False` and are using an NVIDIA processor, try entering `nvidia-smi` at your shell prompt: if you do *not* get a reasonable response, you probably need to troubleshoot your GPU-Driver-Cuda triad -- an ML rite of passage surely sent from the depths of hell. Godspeed.

### Install and test the tensorflow object detection api
Gigadetector uses Google's object detector API. For more on the API see:
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

Test all of the above with the following:

      cd ~/models/research
      python object_detection/builders/model_builder_tf1_test.py

If you see a bunch of lines indicating tests were Run (maybe with one or two tests skipped), you are good to go!

### Download model and test images
The frozen model and test images we use is too large to hold at github, so let's download them:

    cd ~/gigadetector
    python download.py

You should see some indications that you are downloading images and the frozen fish faster rcnn network (`fish_frcnn_graph.pb`).

## Test it out
If everything until now has run smoothly, congrats! That was the hard part. Now the fun bits begin we can start running the model on some data.

### Test on a small image
Test the model on a small test image that contains two fish: this image is large by most ML pipeline standards (1200 x 1200) but very small by gigadetector standards:

    conda activate gigadetector
    cd ~/gigadetector/gigadetector/
    python tiny_test.py

You should see an image pop up with two fish outlined by bounding boxes with the confidence measure (a *score*). Press ESC over the image to close the window.

Note in the rest of the examples the first two lines from above (activating the environment and cd'ing into `~/gigadetector/gigadetector/`) will be assumed.

### Test on a single gigaimage
Here we will have a very "verbose" example that shows a single large image with 93 zebrafish that shows the moving window walking through it, where the faster-rcnn is applied, and that same window is shown on the side with the output of the faster-rcnn (whether it found fish or not):

    python gigatest.py

You will see the algorithm run through an entire single gigaimage with a sliding window (it may run relatively quickly). This also saves the results in `~/gigadetector/data/processed/giga1_od_results.pkl` (including bounding boxes, scores, ROIs of the original image, and the file path to the original image).

Because this sliding window draws *too many* bounding boxes (often multiple per fish), we need to try to narrow them down to one per fish using tricks such as non-max suppression. This is done in the following:

    python bb_extract.py

This generates a final set of bounding boxes and writes them into a final image that is saved as `~/gigadetector/data/processed/processed_images/giga1_bboxes.png` that you can inspect for quality. This script also has many intermediate steps that are not rendered, but the code is there to do so, so it is easy to inspect to how any given image goes from the full set of initial bounding boxes to the final estimate.


### Run through a folder of images
This is where we get to something that is more like what you will actually do with your real data.

    python gigatest_folder.py

This runs the faster-rcnn on each image in `gigadetector/data/` (there are  three we downloaded there by default). It takes 3-5 minutes per image, as they are very large.

The results are saved in `data/processed/gigafolder_od_results.pkl`, and you can narrow down the bounding boxes with:

    python bb_extract_folder.py

This module saves the results in `gigafolder_bb_results.pkl` and the images with bboxes overlaid are in `/gigadetector/data/processed/processed_images`.

### Viewing images with boxes on them
We provide an interface that lets you reconstruct the images with bounding boxes overlaid, given just the original images and the data generated from the bounding box extraction step. It's a very lightweight OpenCV interface where you click `n` to move to the next image in the set and `Esc` to close the viewer:

    python gigaviewer.py

This is helpful to just make sure the bounding boxes match up to the images, and should be enough to get you started with how to integrate OpenCV with gigadetector.

## Applying to new images
If you want to run these algorithms on your own data, then I would adapt the code from the test examples. Start out with a single small (1200x1200) image and make sure you can get it to work. Then make sure you can get it to work on a single large image of your own, and then a folder of images. Finally, build up your own project as you see fit with your own directory structure. Good luck!

If you have any problems/suggestions, please open an issue or PR.

## To do
- pics
- let folks know


### Acknowledgments
Code developed for the multi-camera array microscope (MCAM) project in the Computational Optics Lab (http://horstmeyer.pratt.duke.edu/) and the Naumann Lab (https://www.naumannlab.org/) at Duke University.
