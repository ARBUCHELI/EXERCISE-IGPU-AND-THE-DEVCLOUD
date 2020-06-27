# EXERCISE-IGPU-AND-THE-DEVCLOUD

## This Content was Created by Intel Edge AI for IoT Developers UDACITY Nanodegree.

Requesting a CPU on Intel's DevCloud and loading a model on the IGPU, and running inference on an image using both a CPU and IGPU.

## Exercise: Integrated GPU (IGPU) and the DevCloud

Requesting a CPU on Intel's DevCloud and loading a model on the IGPU, you will have the opportunity to do this yourself with the addition of running inference on an image using
both a CPU and IGPU.

On the previous page, we concluded that running the inference on the IGPU could potentially improve the performance enough that the client would not need to purchase any new 
hardware. However, we don't know for sure if this is the case—we need to test it. So in this exercise, we'll also investigate how batching affects the inference.

In this exercise, you will do the following:

1. Write a Python script to load a model and run inference 1000 times with and without batches.
     * Calculate the time it takes to load the model.
     * Calculate the time it takes to run inference 1000 times with a batch of 1.
     * Calculate the time it takes to run inference 100 times with a batch of 10.
     
2. Write a shell script to submit a job to Intel's DevCloud.
3. Submit two jobs using <code>qsub</code> on an edge node.

    * One job using <code>CPU</code> as the device with an <strong>UP Squared Grove IoT Dev kit</strong>.
    * One job using <code>GPU</code> as the device with an <strong>UP Squared Grove IoT Dev kit</strong>.
    
4. Run <code><liveQStat</code> to view the status of your submitted jobs.
5. Retrieve the results from your job.
6. View the results.
7. Plot and compare the results using bar graphs with <code>matplotlib</code> for the following metrics:
    
    * Model Loading Time
    * Inference Time
    * Frames Per Secone (FPS)
 
<strong>IMPORTANT: Set up paths so we can run Dev Cloud utilities</strong>

You must run this every time you enter a Workspace session.

<pre><code>
%env PATH=/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/intel_devcloud_support
import os
import sys
sys.path.insert(0, os.path.abspath('/opt/intel_devcloud_support'))
sys.path.insert(0, os.path.abspath('/opt/intel'))
</code></pre>

# The Model
We will be using the <code>vehicle-license-plate-detection-barrier-0106</code> model for this exercise.

Remember to use the appropriate model precisions for each device:

* CPU - FP32
* IGPU - FP16
The model has already been downloaded for you in the <code>/data/models/intel directory</code> on Intel's DevCloud.

We will be running inference on an image of a car. The path to the image is <code>/data/resources/car.png</code>.

# Step 1: Creating a Python Script
The first step is to create a Python script that you can use to load the model and perform inference. We'll use the <code>%%writefile</code> magic to create a Python file called <code>inference_on_device.py</code>. In the next cell, you will need to complete the <code>TODO</code> items for this Python script.

<code>TODO</code> items:

1. Load the model
2. Get the name of the input node
3. Prepare the model for inference (create an input dictionary)
4. Run inference 100 times in a loop when the batch size is 10, and 1000 times when the batch size is 1

<pre><code>
%%writefile inference_on_device.py

import time
import numpy as np
import cv2
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import argparse

def main(args):
    model=args.model_path
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    batches=int(args.batches)
    
    start=time.time()
    model=IENetwork(model_structure, model_weights)
    model.batch_size=batches

    core = IECore()
    net = core.load_network(network=model, device_name=args.device, num_requests=1)
    
    # TODO: Load the model
       
    load_time=time.time()-start
    print(f"Time taken to load model = {load_time} seconds")
    
    # TODO: Get the name of the input node
    input_name=next(iter(model.inputs))
    # Reading and Preprocessing Image
    input_img=cv2.imread('/data/resources/car.png')
    input_img=cv2.resize(input_img, (300,300), interpolation = cv2.INTER_AREA)
    input_img=np.moveaxis(input_img, -1, 0)
    
    # Running Inference in a loop on the same image
    input_dict={input_name:[input_img]*batches}
    
    if batches==1:
        iterations=1000
    else:
        iterations=100
    
    start=time.time()
    for _ in range(iterations):
        # TODO: Run Inference in a Loop
        net.infer(input_dict)
    
    # Calculate inference time and fps
    inference_time=time.time()-start
    fps=100/inference_time
    
    print(f"Time Taken to run 100 Inference is = {inference_time} seconds")
    
    # Write load time, inference time, and fps to txt file
    with open(f"/output/{args.path}.txt", "w") as f:
        f.write(str(load_time)+'\n')
        f.write(str(inference_time)+'\n')
        f.write(str(fps)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--path', default=None)
    parser.add_argument('--batches', default=None)
    
    args=parser.parse_args() 
    main(args)
</code></pre>

# Step 2: Creating a Job Submission Script
To submit a job to the DevCloud, you'll need to create a shell script. Similar to the Python script above, we'll use the <code>%%writefile</code> magic command to create a shell script called <code>inference_model_job.sh</code>. In the next cell, you will need to complete the <code>TODO</code> items for this shell script.

<code>TODO</code> items:

1. Create three variables:

    * DEVICE - Assign the value as the first argument passed into the shell script.
    * BATCHES - Assign the value as the second argument passed into the shell script
    * MODELPATH - Assign the value as the third argument passed into the shell script.
    * SAVEPATH - Assign the value as the fourth argument passed into the shell script.
2. Call the Python script using the three variable values as the command line argument.

<pre><code>
%%writefile inference_model_job.sh
#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

# TODO: Create DEVICE variable
# TODO: Create BATCHES variable
# TODO: Create MODELPATH variable
# TODO: Create SAVEPATH variable
DEVICE=$1
BATCHES=$2
MODELPATH=$3
SAVEPATH=$4

# TODO: Call the Python script
python3 inference_on_device.py  --model_path ${MODELPATH} --device ${DEVICE} --path ${SAVEPATH} --batches ${BATCHES}

cd /output

tar zcvf output.tgz * # compresses all files in the current directory (output)
</code></pre>

# Step 3: Submitting a Job to Intel's DevCloud
In the next two sub-steps, you will write your <code>!qsub</code> commands to submit your jobs to Intel's DevCloud to load your model and run inference on the UP Squared Grove IoT Dev kit with an <strong>Intel Atom x7-E3950</strong> CPU and <strong>Intel HD Graphics 505</strong> IGPU.

Your<code>!qsub</code> command should take the following flags and arguments:

1 The first argument should be the shell script filename
2. <code>-d</code> flag - This argument should be <code>.</code>
3. <code>-l</code> flag - This argument should request an edge node with an <strong>UP Squared Grove IoT Dev kit</strong>. The node name for this is <code>up-squared</code>. The kit contains the following devices:

        * Intel Atom x7-E3950 for your <code>CPU</code>
        * Intel HD Graphics 505 for your <code>IGPU</code>
        
<strong>Note</strong>: Since this is a development kit with a predetermined set of hardware, you don't need to specify the CPU and GPU on your node request like we did in previous exercises.

4. <code>-F</code> flag - This argument should contain the three values to assign to the variables of the shell script:

    * <strong>DEVICE</strong> - Device type for the job: <code>CPU</code> or <code>GPU</code>
    * BATCHES - Batch size. <code>1 or 10</code>
    * MODELPATH - Full path to the model for the job. As a reminder, the model is located in <code>/data/models/intel</code>
    * SAVEPATH - Name of the file you want to save the performance metrics as. These should be named as the following:
        * <code>cpu_stats</code> for the <code>CPU</code> job without batching
        * <code>cpu_batch_stats</code> for the <code>CPU</code> job with batching
        * <code>gpu_stats</code> for the <code>GPU</code> job without batching
        * <code>gpu_batch_stats</code> for the <code>GPU</code> job with batching
        
<strong>Note</strong>: There is an optional flag, <code>-N</code>, you may see in a few exercises. This is an argument that only works on Intel's DevCloud that allows you to name your job submission. This argument doesn't work in Udacity's workspace integration with Intel's DevCloud.

# Solution of the exercise and adaptation as a Repository: Andrés R. Bücheli.

