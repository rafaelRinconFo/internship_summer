# **Learning-based Underwater Depth Map Generation**

This final repository has been created in order to organize information in a clearer way. It contains models, script files and experiment notebooks. Other types of files such as depth outputs, image reconstruction ouptuts or trained model files, are contained in the Transcend Disk given to me by Maxime.


# Project goals

:camera: **Images** acquired with Remotely Operated Vehicules provide insightful data. Yet, 2D images only are hardly sufficient to get a general idea of the **scene 3D structure**. **Depth maps** are essential in the 3D reconstruction process. **Deep-Learning** approaches could provide very useful additional data.



# About this repository

## File structure



    ├── Supervised
        └── notebooks
        └── script_files
            └── models
    ├── Self-supervised
        └── notebooks
        └── script_files
            └── models
        └── tensorboard_runs
    ├──Interesting_papers
    ├── README.md
    ├── Depth_estimation_poster.pdf




The results presented here were obtained on a 3526 images dataset, taken on the following sites
- _Training set_ : Old rainbow Cliff+ Thermithiere -> 2845 images  
- _Test set_:  Ghost city+Medium structure, 681 images -> 681 images

## Models
The different models available:

**SUPERVISED APPROACH**
- Midas
- ResNet 50, ImageNet pretraining
- Sparse Autoencoder, ImageNet pretraining
- Feature Fusion, No pret-training step in the encoder part.



**UNSUPERVISED APPROACH**
- Feature fusion based, sparse depth map + RGB as an input.


## Perspectives 

:construction:  What could be done in the future about this subject 
- Improve dataset quality by removing images with haze/ high turbidity/ pose matrix problems
- Test self-supervision with a pre-trained Feature Fusion Network
- Spot pose matrix errors and remove the errors from those images.
- Add the Victor cam images, to build a bigger dataset. Multiple camera model?


## Authors and acknowledgment
Inès Larroche. July-December 2022  
Role models: *Maxime and Clémentin*.   

Special thanks to them :pray: :sparkles:

Contact: ines.larroche@eleves.enpc.fr



 
