# Neural_Nets

Install singularity: https://sylabs.io/guides/3.0/user-guide/index.html

Build a container with the recipe in this repository.

To enable GPU usage, check you have all necessary libraries and prerequisites at: https://www.tensorflow.org/install/gpu

Start playing with Neural Nets and Helix prediction:
1 - download pdbs in pdb_list (there's a bash script in Utilities to download them from command line)
2 - install the dssp software (sudo apt update && sudo apt-get install dssp/bionic)
3 - run dssp on all the downloaded pdb files
4 - modify formatter.py, trainer.py and predictor.py to take the correct paths as input (according to your folder setup)
5 - tweak input lists/network parameters and train your Network!
