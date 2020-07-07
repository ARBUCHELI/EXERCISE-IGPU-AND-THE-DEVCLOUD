# EXERCISE-IGPU-AND-THE-DEVCLOUD

## This Content was Created by Intel Edge AI for IoT Developers UDACITY Nanodegree. (Solution of the exercise and adaptation as a repository: Andrés R. Bucheli.)

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
    * <strong>BATCHES</strong> - Batch size. <code>1 or 10</code>
    * <strong>MODELPATH</strong> - Full path to the model for the job. As a reminder, the model is located in <code>/data/models/intel</code>
    * <strong>SAVEPATH</strong> - Name of the file you want to save the performance metrics as. These should be named as the following:
        * <code>cpu_stats</code> for the <code>CPU</code> job without batching
        * <code>cpu_batch_stats</code> for the <code>CPU</code> job with batching
        * <code>gpu_stats</code> for the <code>GPU</code> job without batching
        * <code>gpu_batch_stats</code> for the <code>GPU</code> job with batching
        
<strong>Note</strong>: There is an optional flag, <code>-N</code>, you may see in a few exercises. This is an argument that only works on Intel's DevCloud that allows you to name your job submission. This argument doesn't work in Udacity's workspace integration with Intel's DevCloud.

## Step 3a: Running on the CPU
In the cell below, write the qsub command that will submit your job to the CPU with a batch size of 1.

<pre><code>
cpu_job_id_core = !qsub inference_model_job.sh -d . -l nodes=1:up-squared -F "CPU 1 /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106 cpu_stats" -N store_core 
print(cpu_job_id_core[0])
</code></pre>

## Step 3b: Running on the CPU with Batches
In the cell below, write the qsub command that will submit your job to the CPU with a batch size of 10.

<pre><code>
cpu_batch_job_id_core = !qsub inference_model_job.sh -d . -l nodes=1:up-squared -F "CPU 10 /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106 cpu_batch_stats" -N store_core 
print(cpu_batch_job_id_core[0])
</code></pre>

## Step 3c: Running on the GPU
In the cell below, write the qsub command that will submit your job to the GPU with a batch size of 1.

<pre><code>
gpu_job_id_core = !qsub inference_model_job.sh -d . -l nodes=1:up-squared -F "GPU 1 /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106 gpu_stats" -N store_core 
print(gpu_job_id_core[0])
</code></pre>

## Step 3d: Running on the GPU with batches
In the cell below, write the qsub command that will submit your job to the GPU with a batch size of 10.

<pre><code>
gpu_batch_job_id_core = !qsub inference_model_job.sh -d . -l nodes=1:up-squared -F "GPU 10 /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106 gpu_batch_stats" -N store_core 
print(gpu_batch_job_id_core[0])
</code></pre>

# Step 4: Running liveQStat

Running the <code>liveQStat</code> function, we can see the live status of our job. Running the this function will lock the cell and poll the job status 10 times. The cell is locked until this finishes polling 10 times or you can interrupt the kernel to stop it by pressing the stop button at the top:

* <code>Q</code> status means our job is currently awaiting an available node
* <code>R</code> status means our job is currently running on the requested node

<strong>Note</strong>: In the demonstration, it is pointed out that <code>W</code> status means your job is done. This is no longer accurate. Once a job has finished running, it will no longer show in the list when running the <code>liveQStat</code> function.

<pre><code>
import liveQStat
liveQStat.liveQStat()
</code></pre>

# Step 5: Retrieving Output Files

In this step, we'll be using the <code>getResults</code> function to retrieve our job's results. This function takes a few arguments.

1. <code>job id</code> - This values are stored in the variables for each job you submitted using the qsub command in <strong>Step 3a, Step 3b, Step 3c</strong>, and 
<strong>Step 3d</strong>:

    * <code>cpu_job_id_core</code>
    * <code>cpu_batch_job_id_core</code>
    * <code>gpu_job_id_core</code>
    * <code>gpu_batch_job_id_core</code>
    
Remember that this value is an array with a single string, so we access the string value using <code>job_id_core[0]</code>.

2. <code>filename</code> - This value should match the filename of the compressed file we have in our <code>inference_model_job.sh</code> shell script.
3. <code>blocking</code> - This is an optional argument and is set to <code>False</code> by default. If this is set to <code>True</code>, the cell is locked while waiting for the results to come back. There is a status indicator showing the cell is waiting on results.

<strong>Note</strong>: The <code>getResults</code> function is unique to Udacity's workspace integration with Intel's DevCloud. When working on Intel's DevCloud environment, your job's results are automatically retrieved and placed in your working directory.

## Step 5a: Get GPU Results
<strong>Without batches</strong>

<pre><code>
import get_results
get_results.getResults(gpu_job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

<pre><code>!tar zxf output.tgz</code></pre>
<pre><code>!cat stdout.log</code></pre>
<pre><code>!cat stderr.log</code></pre>

<strong>With Batches</strong>

<pre><code>
import get_results
get_results.getResults(gpu_batch_job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

<pre><code>!tar zxf output.tgz</code></pre>
<pre><code>!cat stdout.log</code></pre>
<pre><code>!cat stderr.log</code></pre>

## Step 5b: Get CPU Results
<strong>Without Batches</strong>

<pre><code>
import get_results
get_results.getResults(cpu_job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

<pre><code>!tar zxf output.tgz</code></pre>
<pre><code>!cat stdout.log</code></pre>
<pre><code>!cat stderr.log</code></pre>

<strong>With Batches</strong>

<pre><code>
import get_results
get_results.getResults(cpu_batch_job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

<pre><code>!tar zxf output.tgz</code></pre>
<pre><code>!cat stdout.log</code></pre>
<pre><code>!cat stderr.log</code></pre>

# Step 6: View the Outputs

Can you plot the load time, inference time and the frames per second in the cell below?

<pre><code>
import matplotlib.pyplot as plt
</code></pre>

<pre><code>
def plot(labels, data, title, label):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.bar(labels, data)
    
def read_files(paths, labels):
    load_time=[]
    inference_time=[]
    fps=[]
    
    for path in paths:
        if os.path.isfile(path):
            f=open(path, 'r')
            load_time.append(float(f.readline()))
            inference_time.append(float(f.readline()))
            fps.append(float(f.readline()))

    plot(labels, load_time, 'Model Load Time', 'seconds')
    plot(labels, inference_time, 'Inference Time', 'seconds')
    plot(labels, fps, 'Frames per Second', 'Frames')

paths=['gpu_stats.txt', 'cpu_stats.txt', 'gpu_batch_stats.txt', 'cpu_batch_stats.txt']
read_files(paths, ['GPU', 'CPU', 'GPU Batching', 'CPU Batching'])
</code></pre>

![image](https://raw.githubusercontent.com/ARBUCHELI/EXERCISE-IGPU-AND-THE-DEVCLOUD/master/download1.png)
![image](https://raw.githubusercontent.com/ARBUCHELI/EXERCISE-IGPU-AND-THE-DEVCLOUD/master/download2.png)
![image](https://raw.githubusercontent.com/ARBUCHELI/EXERCISE-IGPU-AND-THE-DEVCLOUD/master/download3.png)
# Conclusion
We can see that batching the images leads to some improvement in <strong>inference time</strong> and <strong>FPS</strong> for both the CPU and GPU; however, we can see the improvement in performance for the GPU is much better.



## Solution of the exercise and adaptation as a Repository: Andrés R. Bucheli.

