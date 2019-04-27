# Introduction
Trajectory optimization code following Matthew Kelly's tutorial here: http://www.matthewpeterkelly.com/research/MatthewKelly_IntroTrajectoryOptimization_SIAM_Review_2017.pdf

This repo contains some simple transcriptions of trajectory optimization problems to evaluate how well it works. Note, the implementations are not fast and are merely for testing.

# Organization
The first file to look at is *block_move_trapezoidal.py* which contains functions for a simple double integrator problem that have been transcribed by hand. The functions are not generalizable, and is just there to give a simple example of a transcription of an easy problem. It is easy to debug and see the program flow.

Next, a more general implementation of transcription is in *transcription.py* which contains a general class that takes in a continuous problem and spits back the transcribed version. It works with multi dimensional state and control as well as both trapezoidal and hermite-simpson integration approximations.

To test *transcription.py*, there is a file *plot_trajectories.py* which will can solve a double integrator or pendulum swing up problem and plot the resulting trajectories.
In addition, to test how errors scale with different different integration approximations and number of knot points, there is a file *test_errors.py*

# Dependencies
* python3
* numpy
* scipy
* matplotlib
* argparse


