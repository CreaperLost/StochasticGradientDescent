Setup your Python environment to run the assignment code using PYTHON 2.7 !!!!

[Option 1] Use Anaconda 

The preferred approach for installing all the assignment dependencies is to use Anaconda, 
which is a Python distribution that includes many of the most popular Python packages for 
science, math, engineering and data analysis. 
Once you install it you can skip all mentions of requirements and youâ€™re ready to go directly to working on the assignment.

You need to work using Python 2.7 for this assignment.
So using Anaconda environment (whatever version 4.x) it is easy to create an environment to use with Python2.7.

Two steps: 
1. Create the working environment.
>>cd cs587_assignment1
>>conda create --name assign1 python=2.7
>>activate assign1
>>jupyter notebook

If you have problems i.e jupyter notebook is not initiated because of "module not recognized/ module not found" error
or in case you get errors while executing any of the given python scripts/modules in the code then:

Navigate (cd) with the anaconda prompt (found in your start menu in Windows) to your working/assignment folder 
activated the environment you have created using 'activate env_name' and run the following command 

2. Install the required packages 
Method A (all at once):
>>conda install --yes --file requirements.txt

Method B (if A has an issue):
Install only the necessary:
1. Numpy 
2. matplotlib
and whatever else pops up as uninstalled (do not rely on requirements.txt as there might be an issue with your machine and the version that we request). Anaconda will automatically find the compatible and latest versions!


[Option 2] 
Manual install, virtual environment using Python 2.7 distro installed in your machine: 

If youâ€™d like to (instead of Anaconda) go with a more manual and risky installation route you will likely want to create a virtual environment for the project. 
If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. 

To set up a virtual environment, run the following (the requirements.txt file in located in the assignment1 folder).

cd assignment1
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment

The following link might be helpful http://stackoverflow.com/questions/5506110/is-it-possible-to-install-another-version-of-python-to-virtualenv