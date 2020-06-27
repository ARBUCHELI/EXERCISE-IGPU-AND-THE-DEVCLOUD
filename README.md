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
    . Calculate the time it takes to load the model.
    . Calculate the time it takes to run inference 1000 times with a batch of 1.
    . Calculate the time it takes to run inference 100 times with a batch of 10.
2. Write a shell script to submit a job to Intel's DevCloud.
3. Submit two jobs using <code>qsub</code> on an edge node.

    . One job using <code>CPU</code> as the device with an <strong>UP Squared Grove IoT Dev kit</strong>.
    . One job using <code>GPU</code> as the device with an <strong>UP Squared Grove IoT Dev kit</strong>.
4. Run <code><liveQStat</code> to view the status of your submitted jobs.
5. Retrieve the results from your job.
6. View the results.
    7. Plot and compare the results using bar graphs with <code>matplotlib</code> for the following metrics:
    
    . Model Loading Time
    . Inference Time
    . Frames Per Secone (FPS)

# Solution of the exercise and adaptation as a Repository: Andrés R. Bücheli.

