# hfc
Hider-Finder-Combiner architecture

create a virtual environment using your favourite method and then

```
pip3 install torch torchvision torchaudio lightning speechbrain wandb hydra-core hydra-submitit-launcher hydra-optuna-sweeper pyworld librosa holoviews panel bokeh sysrsync tensorboardX speechbrain --upgrade
pip install sqlalchemy==1.4.46 # bug with optuna breaks newest sqalchemy
```

## TODO tomorrow
sort out validation to log hidden and mels and work real nice
TODO pretrain individual models?
TODO look up sota image to image, speaker ID models, conditional spec/image gen models?

work on a CSTR machine? -- DONE
align val set for hifigan and me --- DONE
log lr
