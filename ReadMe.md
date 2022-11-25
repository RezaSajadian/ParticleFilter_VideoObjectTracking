# Particle Filtering for Object Tracking #

This mini project is a custom Object Tracker which starts by clicking on your desired object to track,
and then the Particle Filter tracks the object, using the visual features of the object which is precisely the pixel color of the mouse/touchpad clicked point.

## configuration file and parameters ##

The configuration file contains the model's parameters for the video and of the particle filter, so that there's no need to touch the object's code to change the behaviour of the filter.

## Requirements ##
To run this particle filter, you need to install OpenCV first.
Please beware that for a more smooth try of this filter, use Virtual environments,
which are in general a far better practice than installing all the packages in the general variable environment of your pc, and pollute all the versions together.

This readme is not intended to extend your knowledge about virtual environments, but for the information about using Virtual Environments, I will suggest taking a look at the [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) project which is a great way to manage your virtual environments.

Also, for installing new python packages, you would need a python package manager,
whihc would be preferably [Pip](https://pypi.org/project/pip/)

after creating and activating your virtual environment, you can install OpenCV
by the following command:     pip install opencv-python

