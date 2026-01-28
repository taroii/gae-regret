# GAE Regret

Repo for BruinML project studying regret bounds for GAE? 

Currently trying to reproduce results from [this GAE Paper](https://arxiv.org/pdf/1506.02438) by Schulman et al.

# Set up

```
conda create -n gae python=3.11
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 
pip install -r requirements.txt
```

# Run experiments

```
python -m experiments.cart_pole_experiment
```