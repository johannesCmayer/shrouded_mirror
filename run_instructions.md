# Step by Step Run Guide
This guide describes how to run my bachelor project.

## Required Software
Install the following software:
- Unity 2018.3.0f2
 - https://unity3d.com/get-unity/download/archive
- Anaconda (python 3.7.0 was used -- any 3.6 should also work)
 - https://www.anaconda.com/distribution/

## Install Python Libaries
Open the Anaconda prompt (installed with anaconda).
Navigate in the prompt to the project root (the directory that contains this file) using the cd command.
Then create an environment with conda using the environment.yml file.
```
conda env create -f environment.yml
```

## Execute Programm
In the anaconda prompt, that has the working directory set to the project root, type the following commands to start up the python side.
```
conda activate gqn
python -m SimpleGQN.SimpleGQNMain
```

Load the project into unity and open the scene "Maze_3_gridCapturePointCapture" (found in the Assets folder), and press the play button.
