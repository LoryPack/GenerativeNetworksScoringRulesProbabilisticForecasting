# GenerativeNetworksScoringRulesProbabilisticForecasting
 
We provide here code to run forecast experiments on: 

- y coordinate of 3-dimensional Lorenz63 model.
- 8-dimensional x variables for the Lorenz96 model integrated using the parametrized model.
- WeatherBench; we use the Z500 variable at the coarsest resolution (32x64). See [here](https://github.com/pangeo-data/WeatherBench) for Download instructions


# Reproducibility: 

In the `config` folder, we report `.json` files containing the arguments used to run the experiments reported in the paper.

# Scripts
We have 5 scripts: 

- `generate_data.py` needs to be run to generate datasets for Lorenz63 and Lorenz96
- `train_nn.py` trains the generative networks with the different methods
- `predict_test_plot.py` computes performance metrics and creates plots
- `predict_test_plot_comparison.py` creates comparison plots between three selected methods, for Lorenz63 and Lorenz96
- `plot_weatherbench.py` creates the plots for WeatherBench data


# Dependencies
## Pip 

The dependencies can be installed with ```pip install -r requirements.txt```. 
However that is not enough to use the plotting features (cartopy). for that need to use Conda, see below.

## Conda

```
conda create --name <env-name>
conda install --file requirements_conda.txt
```

but also need to install two other packages from pip: 

```pip install torchtyping typeguard einops```
 
If you want to use GPU, Pytorch has to be installed with the following, instead of the above  
```conda install pytorch cudatoolkit=10.2 -c pytorch```





