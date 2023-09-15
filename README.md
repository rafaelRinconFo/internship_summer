# Repository created for the 2023 summer internship in collaboration between LIS Lab and Ifremer


## General context

This project makes use of two main models. For the supervised baseline, the [MIDAS Pytorch implementation](https://pytorch.org/hub/intelisl_midas_v2/) is used and tuned with the data of the [Eiffel Tower: A Deep-Sea Underwater Dataset for Long-Term Visual Localization](https://www.seanoe.org/data/00810/92226/) dataset. For the unsupervised approach, the architecture in [Unsupervised Monocular Depth Learning in Dynamic Scenes](https://arxiv.org/pdf/2010.16404.pdf), an article from Google research which tries to address the problem of both camera pose estimation and depth estimation with monocular cameras.

The main idea is to generate a supervised baseline which in theory will have more stable results and will need less data to achieve good results. Then, its results will be compared with the non-supervised approach. In addition, the weights of the supervise model can also be used as the base of the depth estimation part of the unsupervised approach.

## Setting up the environment

For starting just set up the virtual environment of the project. It is called `unsupervised_monocular` and you only need to execute:

```
conda env create -f environment.yml
```

Activate it just to be sure that everything is alright with:

```
conda activate unsupervised_monocular
```
Now that the environment is set and ready, it is necessary to activate the submodule included in the repository. For this, just use the following command:

```
git submodule update --init --recursive
```

The submodule is located in the [depth_map_2_mesh_ray_tracer](https://github.com/ferreram/depth_map_2_mesh_ray_tracer) folder. Follow the instructions included in it to compile it, you will need it later. 

**Hint:** If you have issued during the building process, it could be related either to the OpenCV dynamic libraries or the fix mentioned in this [github issue](https://github.com/ferreram/depth_map_2_mesh_ray_tracer/issues/3). At least those were the ones if found during my work.

Once the environment and the submodule are set and ready, you can proceed to download the data.

## COLMAP

This repository assumes that you have already installed [Colmap](https://colmap.github.io/) or that the environment where you are working has already a pre-installed instance of it. If you're working in the LisLAB cluster, it's likely that you already have access to the [Colmap](https://colmap.github.io/) tools by just adding them to your cluster instance. If that's not the case, you have to install it on your local machine by following the [Colmap installation guide and tutorial](https://colmap.github.io/install.html).

## Data

You need to know a couple of things about the dataset before downloading it. First of all, the dataset that the project was designed to use is the [Eiffel Tower: A Deep-Sea Underwater Dataset for Long-Term Visual Localization](https://www.seanoe.org/data/00810/92226/). This dataset was taken according to the [Colmap](https://colmap.github.io/) format (I strongly suggest you read about the format and its structure). 

### Downloading the dataset

Inside the `scripts` folder there's a script for downloading one of the years included in the dataset. Up to this point (September 2023) there are 4 years composing the entire dataset: 2015, 2016, 2018, and 2020. The script allows you to download one year at a time or several years if you want. Just keep in mind that every year has around 10Gb of data, data that must be downloaded and decompressed. You can try with only one year in your local machine and if you need to download the entire set of years, a cluster-alike environment is highly recommended. 

The command should look something like this where XXXX or YYYY stand for the year(s) to be downloaded:

```
python3 scripts/download_dataset.py --years XXXX YYYY
```

As mentioned before, the script will decompress the data and delete the `.zip` files. By default, all data will be store in the following format:

```bash
internship_summer/datasets/
└── XXXX
└── YYYY

```
By default, the datasets are stored in the `datasets` folder.

Again, keep in mind that it is necessary to have a considerable amount of disk space before performing this step.




### Obtaining the ground truth data

You may be wondering: "Why do I need such thing as the ground truth if I am going to train an unsupervised method?". Well, it's a fair question. The answer is pretty simple, one of the ideas of the project is to compare the performance in general of the unsupervised method vs. already existing and well-tested supervised methods. 

In order to do so, the ground truth must be obtained with the help of the already mentioned Colmap and the *depth_map_2_mesh_ray_tracer*.

#### Getting the mesh

First, you must obtain the mesh from the images and poses provided by the dataset. In order to do so, there's a bash script inside the `bash_scripts` folder called [obtain_mesh.sh](/bash_scripts/obtain_mesh.sh). This script is designed for obtaining the mesh within the Cluster environment of the LisLAB but the commands should be the same if you want to obtain a mesh in your private NASA computer. 

The process has 4 steps. Every single one of those steps is better explained in the Colmap documentation but I will mention them for you here:

* Obtaining the undistorted images. 
* Executing the patch-match-stereo-process.
* Perform the stereo-fusion process.
* Creating the mesh.

You just need to execute the following command inside the `bash_scripts` folder:

```
bash obtain_mesh.sh XXXX
```

Where XXXX is the year from your dataset you want to obtain the mesh from. In the end, you should obtain something like this:

```bash
internship_summer/datasets/
└── XXXX
    ├── dense
    │   ├── fused.ply 
    │   ├── meshed-poisson.ply
```
Where the one we are interested in is the ```meshed-poisson.ply```. You can visualize it with some mesh visualization/manipulation software such as Blender and it should look something like this:

![Mesh view](https://i.imgur.com/R8ILHwP.png)

**NOTE:** The reason why the process takes only one folder at a time is because this is quite a heavy task that takes a bunch of resources of the machine you're working in. It is possible that for the bigger datasets, the task is just going to stop and crash. In that case, you must execute again from the last step you tried to execute and the task should restart from that point and on. It's a long and tedious process, so be patient with this step.


#### Getting the depth images
Once you have obtained the `meshed-poisson.ply` file for the desired dataset, you can execute the command included in the **depth_map_2_mesh_ray_tracer** submodule docs. As a destination folder, you should use `datasets/XXXX/dense/depth` as shown below. If the previous step was successfully executed, you shouldn't have any issue with this step, and all the **.png** images with the depth data will be stored in that folder. 

```bash
internship_summer/datasets/
└── 2020
    ├── dense
    │   ├── depth
```
### Splitting the dataset

By this point you should have all the depth images of your dataset and you are ready to train. First, you need to split your dataset. To do so, you need to execute the following command as follows:

```
python -m scripts.split_dataset --years XXXX YYYY
```

There are two types of splitting: 

* Shuffled: Randomly shuffles all the images and from this, creates the training, validation, and test subsets. As you can imagine, this random shuffle doesn't guarantee any order in the splitting.
* Sequential: In this case, the script will divide the data into sequences of a given size, from this portion of the data, the train, validation, and test subsets will be obtained in the same order as the original dataset. I.e. From the first 1000 images of the dataset, the first 700 will be used for training, the following 200 will be used for validation and the last 100 for test. All of them preserve the order or the initial dataset. There is no shuffle of any kind in this method.

You can specify the type of split as follows. By default, the split is set to shuffle:

```
python -m scripts.split_dataset --years XXXX YYYY --split_type sequential
```

A `.csv` file will be created in the `datasets` directory containing the created split. This `.csv` file will be used by the train scripts.

## Training time!

At this point you should have:

* The images from the original dataset.
* The depth images generated with the ray tracer tool.
* The `.csv` files contain the split of the dataset.

If that's the case, you can now proceed to train, but first, let's have some things in mind:

### Config files

When the time comes to iterate and tune your hyperparameters, you'll need to modify their values in the configuration files located in the `configs` folder. There you will be able to find different kinds of parameters for the [supervised](/configs/supervised_params.yml) and [unsupervised](/configs/unsupervised_params.yml) models. 

As you may notice, the unsupervised config file has way more parameters than the supervised one, this is because the [Unsupervised Monocular Depth Learning in Dynamic Scenes](https://arxiv.org/pdf/2010.16404.pdf) paper proposes way more hyper-tuning options and loss functions than what we do in the tuning of the MIDAS model. It's highly recommended (A polite way of saying you **must**) to read the paper before starting with this project.

Among other things, in the config files you can configure the type of model you want to use, the loss functions of the supervised model, the metrics you want to use, and other kinds of stuff. Check it before you start with the training, it's worth it.

### Visualizing the results

The results of the experiments are displayed in weights and biases. This will allow you to visualize the results of the different epochs while the model is being trained, share it with your colleagues, and to compare the results of different models after the training.

![results](https://i.imgur.com/7uUwchw.png)

If you run experiments in your local NASA supercomputer, run this command in order to enable the log of the metrics in weights and biases.

```
export WANDB_API_KEY=XXXXXXXXX
```

Where `XXXXXXXXX` is the API key of your personal account provided by WANDB.

If you deploy your experiments in the cluster, you need to go to either the [supervised](/bash_scripts/deploy_supervised_job.sh) or [unsupervised](/bash_scripts/deploy_unsupervised_job.sh) according to the experiment you run. There is a line at the very beginning of both scripts where you will be able to include the API key.

What happens if you don't put your API key? Well, absolutely nothing, the code won't break or anything like that. It will just run without displaying the results, which is OK I suppose but not very useful for comparing metrics. So, be careful with this.

### Storing the models:

There is a folder called `experiments`, in this folder you will find the weights of the model you just run. Every experiment is stored within its own folder according to the date it was created. If you want to change the amount frequency used for saving the model, in both config files you can find a parameter called `save_model_every`, this will change the number of epochs before saving the new weights of the model.

### Supervised

If until this point you have followed all the steps, you are ready to execute your first run. I know this was super easy. For executing the supervised training in your local machine, you only need to execute:

```
python -m supervised.train 
```

If you want to run a small experiment just with a small percentage of the total dataset, you can use the toy flag of the script. This will be useful for debugging and quick checks if you just made a small modification:

```
python -m supervised.train --toy true
```

Now, if you want to execute the experiment with the full power of the mighty cluster, there's already a script for it, [the deploy script](/bash_scripts/deploy_supervised_job.sh). If you're working in the LisLAB cluster, you only need to execute:

```
sbatch deploy_supervised_job.sh
```

Just like that, you're running your first experiment.

### Unsupervised
In a very similar way to the supervised case.  To execute the unsupervised training in your local machine, you only need to execute:


```
python -m unsupervised.train 
```

If you want to run a small experiment just with a small percentage of the total dataset, you can use the toy flag of the script. 

```
python -m unsupervised.train --toy true
```

Again, if you want to use the mighty cluster, [the unsupervised deploy script](/bash_scripts/deploy_unsupervised_job.sh) will help you. If you're working in the LisLAB cluster:

```
sbatch deploy_supervised_job.sh
```

## Modifying the code

Do you want to modify the code?? Of course, go for it! Keep in mind a couple of things:

### Modifying the supervised-related scripts

When it comes to the supervised approach, you can modify mainly the metrics. Consider that the MIDAS model is already implemented by Pytorch so there's not a lot to modify regarding the architecture of the model. Now, if you want to add a new model, you totally can do it. I did it with the depth_estimation section of the unsupervised pipeline and it worked very well. Just remember to add it to the config files and the [supervised models script](/supervised/model.py). Be very careful with the input and output shape of the model.

You can modify the losses and metrics or create new ones. Again, just remember to add them to the config file.

### Modifying the unsupervised related scripts

Again, you can modify it as you wish. My only suggestion is to modify only when you have a deep understanding of the paper I mentioned at the beginning. If you check the [losses folder](/unsupervised/losses/) you'll note that there are several scripts for the losses. Why? Well, when we talk about an unsupervised approach every loss helps the model to go in the direction we want. Again, it is vital to read the paper before going to the code.
