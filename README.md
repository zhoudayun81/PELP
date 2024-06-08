# PELP: Prioneer Event Log Prediction

***|-/code*** folder contains all code for all kinds of experiments.
You can modify the parameters in ***_config.ini_*** to run the experiments in different settings.

***|-/input*** folder contains all test inputs for our experiments.
The code will only run the files in the root folder of */input* folder, any files in its subfolders will NOT be run. If you want to run the synthetic input dataset, please move it to the root of the */input* folder.

Feel free to explore the code and input your self! :D

The code runs on Python 3 and with PyTorch library.

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
* Please do not change the name of the */code* and */input* folders.

If you need help or more explanation, please [Contact Us](mailto:w.zhou26@student.unimelb.edu.au) for assistance.
