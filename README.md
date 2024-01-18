# hfc
Hider-Finder-Combiner architecture

create a virtual environment using your favourite method and then

```
pip3 install torch torchvision torchaudio lightning speechbrain wandb hydra-core hydra-submitit-launcher hydra-optuna-sweeper pyworld librosa holoviews panel bokeh sysrsync --upgrade
pip install sqlalchemy==1.4.46 # bug with optuna breaks newest sqalchemy
```

TODO tomorrow
Get rsync working for data if configured to use it. #
everything running on cluster
scheduling working #
run a sweep with wandb etc 