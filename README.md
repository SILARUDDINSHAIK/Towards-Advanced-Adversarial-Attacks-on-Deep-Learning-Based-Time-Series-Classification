
***Towards Advanced Adversarial Attacks on Deep Learning Based Time Series Classification***

*Project Overview*

Deep  learning-based  Time series classification models  are  found  to  be  vulnerable  to  adversarial  attacks, i.e., strategically perturbed examples from test set that cause the model to produce incorrect predictions. In this project we successfully  leverage  Project  Gradient  Descent  Attack and we explored a different way to generate non-gradient based attack by utilizing the similarity among the time series known as distance based attack.  We also adopt the idea from computer vision and build the first TSC model with certified robustness against any L2 adversarial attack.  Our work is evaluated using benchmark TSC datasets from 2018 UCR TSarchive. This repo contains code and trained models for the above experiments.
 
*How to run experiments*

** To Train Base classifiers or Fine tune Deep Learning Models on Time Series Classification tasks**
* In our experiments we had chosen models like Resnet,MLP,FCN.
* Clone the repo navigate to Adversarial_Attacks_On_TSC folder.  
* To train resnet model on TSC dataset: python main.py --device=cpu --model=resnet --dataset=UCR_Coffee
* To train MLP model on TSC dataset: python main.py --device=cpu --model=MLP --dataset=UCR_Coffee
* To train FCN model on TSC dataset: python main.py --device=cpu --model=FCN --dataset=UCR_Coffee

Results will be saved to ./results/DATASET_NAME/ARCHITECTURE_NAME/ by default and the checkpoints can be found there.

** To run Adversarial Attacks on TSC models **

*  To run PGD attack on resnet model: python main.py --device=cpu --model=resnet --dataset=UCR_Coffee --resume --adversarial-eval
*  To run PGD attack on MLP model: python main.py --device=cpu --model=MLP --dataset=UCR_Coffee --resume --adversarial-eval
*  To run PGD attack on FCN model: python main.py --device=cpu --model=FCN --dataset=UCR_Coffee --resume --adversarial-eval
*  To run Distance based attack on selected dataset: DistanceBasedAttack/visualize.py 

** To run Gaussian Noise Attack , Randomized Smoothing technique on TSC models **

* Navigate to Towards_Advance_Adversarial_Attacks_Against_TSC_Adversarial Attacks folder
* To generate gaussian noise and perform randomized smoothing technique Run: certified_accuracy.py
* To change base classifier in args change model ='FCN' or 'MLP' or 'resnet'
* To change dataset update dataset list values to desired dataset names.

To perform additional attacks check Adversarial_Attacks_On_TSC/configs folder.




