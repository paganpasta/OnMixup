


#### Getting Started

#### Pre-requisite

Install [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

#### Building & Running the docker image

1. `nvidia-docker build --rm --tag on_mixup:latest .`
2. `nvidia-docker run -p 8000:8000 -v $PATH/TO/PROJECT_DIR:/home/on_mixup --rm --name on_mixup -it on_mixup:latest`

#### Generating The Density Plots

Firstly, start the jupyter notebook

`jupyter notebook --ip=0.0.0.0 --port=8000 --allow-root --NotebookApp.token='' --NotebookApp.password=''`

There are two notebooks corrsponding to scenarios of Mixup and **no** Mixup. Details on how to run them are provided in the notebooks itself. 

#### Training 

There are files named:

1. cifar.py
2. fmnist.py
3. stl_10.py 

To the train the neural network on the particular dataset execute the corresponding script, for example,
`python cifar.py`


### Evaluation

`evaluation.py` takes as an argument a config.yaml file. Sample configs are in the `eval_config` folder. To run an evaluation on the dataset, execute the following command
`python evaluation.py --config eval_config/cifar.yaml`
The fields of the config are self explanatory. 


  
