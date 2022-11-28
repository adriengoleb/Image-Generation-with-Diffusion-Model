# no-name-idea-fusion for Diffusion Models Project

Adrien Golebiewski - Oumaıma Bendramez - Yann Trémenbert

## Structure:

- *models*: directory containing the model used for generating the 1024 samples
- *notebooks*: directory containing jupyter notebook used during development
- *samples*: directory containing the 1024 generated samples (not run here because my machine is very slow, but it should work if you try!)
- *model.py*: python file describing the Unet model used
- *params.py*: python file containing the global variable used for inference
- *utils.py*: python file with useful utilitary functions
- *generate.py*: python file callable for generating 1024 images
- **Diffusion_Model_Adrien_Oumaima_Yann-2.pdf**: PDF slides for the intermediary presentation
- **Report_Diffusion_Models_Adrien_Oumaima_Yann.pdf**: PDF report (most of the report's figures can be found in the notebook Variance_Schedule_Analysis.ipynb)

## Environment:

- python 3.10.6
- torch 1.13.0
- torchvision 0.14.0
- numpy 1.24.3

## Generation:

> python generate.py
