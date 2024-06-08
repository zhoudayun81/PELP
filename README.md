# PELP: Prioneer Event Log Prediction

|-/code folder contains all code for all kinds of experiments.
You can modify ***_config.ini_*** parameters to run the experiments in different settings.

|-/input folder contains all test inputs for our experiments.
The code will only run the files in the root folder of /input folder, any files in its subfolders will not be run. If you want to run the synthetic input dataset, please move it to the root of the /input folder.

* Please do not change the name of the /code and /input folders.

Feel free to explore the code and input your self! :D

The code will run on Python 3 and with PyTorch library.

Please install the necessary packages through command: 
```console
$ pip install numpy torch
```
You can run command: 
```console
$ python main.py
```
to start running the experiment.

* Please note that the code runs multi-threads, change the parameter `threads = ` under `[Program]` part to a number that is suitable to the computer you are using.
