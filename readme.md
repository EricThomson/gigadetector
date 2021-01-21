# Gigadetector
Object detection pipeline when your images are way too large to fit into GPU RAM. If you have a model that is trained up on objects, but your images are way big for RAM.

<describe basic process here>

## Installation
1. Clone the repo and create the virtual environment
Make sure you have a GPU driver installed. This tends to be the main hangup. On Linux, I recommend using the proprietary drivers for Nvidia cards. Go to the directory that you want to contain the `gigadetector` folder:
   git clone <give html here>
   cd gigadetector
   conda env create -f environment.yml
   conda activate gigadetector
The above command will create the gigadetector virtual environment and activate it.  Note that creating this virtual environment can take several minutes as it installs large packages such as `tensorflow-gpu` and `opencv`.

2. Test your installation of tensorflow
From the command line, go into python, import tensorflow, and see if you have a gpu available:
    python
    import tensorflow as tf
    tf.test.is_gpu_available()
    exit()
If you have a working version of tensorflow-gpu, and a working driver and GPU installed, You should get a bunch of stuff that looks like gibberish, with the final line `True`. If not, try entering `nvidia-smi` at the OS command line: if you do not get a reasonable response from nvidia, you probably need to troubleshoot the GPU-Driver-Cuda triad which is an ML rite of passage sent from the depths of hell.

3. Test the workflow on a sample image
<instructions on how to run on a simple mcam image we have tucked away somewhere>


## Running the code


### Acknowledgments
Code developed as part of the multi-camera array microscope project in the Computational Optics Lab (http://horstmeyer.pratt.duke.edu/) and the Naumann Lab (https://www.naumannlab.org/) at Duke University.
