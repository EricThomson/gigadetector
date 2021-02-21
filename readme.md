# Gigadetector
Object detection pipeline when your images are too big to fit into your GPU. This readme walks you through installation and running simple examples using a frozen model for fish detection. The best way to use it is to adapt the examples for use with your own data. You don't have to use the fish model -- you should be able to download your own frozen model and use this pipeline for any objects.

For details about this pipeline, see [Paper].

## Installation
Note this was created/tested on Linux. Someone might be able to get it to work on a Mac, but no guarantees, and it almost certainly will not work on Windows.

### Create  virtual environment
Clone the repository and create a virtual environment (we'll call the virtual environment `gigadetector` but feel free to rename):

    cd ~
    git clone https://github.com/EricThomson/gigadetector.git
    cd gigadetector
    conda env create -f environment.yml
    conda activate gigadetector
    pip uninstall opencv-python

The above command will create the gigadetector virtual environment and activate it (and uninstall a superfluous version of opencv).  Note creating this virtual environment typically takes several minutes.

### Test tensorflow installation
From the command line, go into python, import tensorflow, and make sure it sees your gpu:

    python
    import tensorflow as tf
    tf.test.is_gpu_available()
    exit()

If you have a working version of `tensorflow-gpu` that sees your gpu, you should get a bunch of stuff that looks like gibberish, with the final line `True`.

If you get `False` try entering `nvidia-smi` at your shell prompt: if you do *not* get a reasonable response, you probably need to troubleshoot the GPU-Driver-Cuda triad -- an ML rite of passage sent from the depths of hell. Good luck.

### Install/test tensorflow object detection api
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

Test all of the above with the following:

      cd ~/models/research
      python object_detection/builders/model_builder_tf1_test.py

If you see a bunch of lines indicating tests were Run (maybe with one or two tests skipped), you are good to go!

### Download the gigadetector model and test images
Run the script to download the frozen model and download a few large images to test it with.

    cd ~/gigadetector
    python download.py

You should see some indications that you are downloading images and the frozen fish faster rcnn network (`fish_frcnn_graph.pb`).

## Test it out!
If everything until now has run smoothly, congrats! That was the hard part. Now the fun bits begin we can start running the model on some data.

### Test on small image
Test the model on a small test image that contains two fish: this image is large by most ML pipeline standards (1200 x 1200 but very small by gigadetector standards):

    conda activate gigadetector
    cd ~/gigadetector/gigadetector/
    python tiny_test.py

You might see some warnings, but you should see an image pop up with two fish outlined with the confidence measure. Press ESC over the image to close the window.

Note in the rest of the examples the first two lines from above (activating the environment and cd'ing into `~/gigadetector/gigadetector/`) will be assumed.

### Test on a single gigaimage
Here we will have a very verbose example that shows a single large image with 93 zebrafish that shows the moving window walking through it, where the faster-rcnn is applied, and that same window is shown on the side with the output of the faster-rcnn (whether it found fish or not):

    python gigatest.py

You will see the algorithm run through an entire single gigaimage with a sliding window (it may run relatively quickly). This also saves the results in `~/gigadetector/data/processed/giga1_od_results.pkl` (including bounding boxes, scores, ROIs of the original image, and the file path to the original image).

Because this sliding window draws *too many* bounding boxes (often multiple per fish), we need to try to narrow them down to one per fish using tricks such as non-max suppression. This is done in the following:

    python bb_extract.py

This generates a final set of bounding boxes and writes them into a final image that is saved as `~/gigadetector/data/processed/processed_images/giga1_bboxes.png` that you can inspect for quality. This script also has many intermediate steps that are not rendered, but the code is there to do so, so it is easy to inspect to how any given image goes from the full set of initial bounding boxes to the final estimate.


### Run through a folder of images
This is where we get to something that is more like what you will actually do with your real data.

    python gigatest_folder.py

This runs the faster-rcnn on each image in `gigadetector/data/` (there are only three in there by default). It takes 3-5 minutes per image, as they are very large. It saves the results in `gigafolder_od_results.pkl` Then you can narrow down the bounding boxes with:

    python bb_extract_folder.py

This saves the results in `gigafolder_bb_results.pkl` and the images with bboxes overlaid in `/gigadetector/data/processed/processed_images`.

### Viewing images with boxes on them
We have a *very* interface that lets you reconstruct the images with bounding boxes given just the original images and the data generated. This uses a lightweight OpenCV viewer interface where you click `n` to move to the next image in the set to view bounding boxes, and `Esc` to close the viewer:

    python gigaviewer.py

This is helpful to just make sure the bounding boxes match up to the images.

## Next steps
If you want to run these algorithms on your own data, then I would adapt the code from the last examples. Start out with a single image and make sure you can get it to work. Then make sure you can get it to work on a single gigaimage of your own. Finally, start cycling through images in a folder.

## To do
- add a picture of fish with all the fish then reduced boxes and make this part of main page, don't worry about plotting it (you can tell people they can uncomment etc if they want).
- let folks know


### Acknowledgments
Code developed for the multi-camera array microscope (MCAM) project in the Computational Optics Lab (http://horstmeyer.pratt.duke.edu/) and the Naumann Lab (https://www.naumannlab.org/) at Duke University.
