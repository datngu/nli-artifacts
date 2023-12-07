# Dissecting vocabulary biases in Natural Language Inference datasets through Statistical testing approach and Automated data augmentation for artifact mitigation



## Getting Started
You'll need Python >= 3.6 to run the code in this repo.

First, clone the repository:

`git clone git@github.com:datngu/nli-artifacts.git`

and follow the README.md file in the __run__ directory to install the base version.

You then need to install: __nlpaug__ package from: https://github.com/makcedward/nlpaug
Please check its homepage to install it properly.


## Obtain the baseline dataset

Please follow steps by steps in the notebook of __data_writer.ipynb__ to write the original dataset and generate the hypothesis-only data.

## Statistical testing

Implemtation of the poposed statistical test can be done by running the __statistical_test.ipynb__.

Related figures and data are automatically generated.

### Note:

Please make sure that related packages are installed.


## Obtain augmented data

We prepare customized python and bash scripts in the subdirectory of aug to generate the augmented data.

Please remember to download the related models/databases if you like to use sysnomyn and embbeding augmentation. 

Links and instructions to download these data are provided in the python script.

## Training and optimizing the models

In the __run__ subdirectory, we provided python scripts for training and optimize the ELECTRA small model.
We also provided slurm scripts that we used to run the experiments in the __run_experiments__ directory.




