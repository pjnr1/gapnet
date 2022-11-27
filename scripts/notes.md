# Notes

A collection of small notes and usage tips for the scripts in this folder.

The whole library is designed to be run from the project base folder, i.e. any script should be called from there:

```shell
python scripts/spikeogramorator.py ...
```

## `spikeogramorator`

This script generates torch-tensors from AN-simulations.


Basic usage:

```shell
python3 scripts/spikeogramorator.py -m 2d -r -w 1e-3 -j $(lscpu | grep "^CPU(s)" | tr -dc '0-9') -i <an_simulation_folder> -o <output_folder>
```

with the flag `-r`, the script is called in recursive mode and will by default look for any `.mat`-file in the folder
following `-i`. If you have multiple folders for different experiments, simply just replace `<an_simulation_folder>` with
the parent folder that contains all AN-simulations.


Example; used on the cluster:
```shell
python3 scripts/spikeogramorator.py -m 2d -r -w 1e-3 -j $(lscpu | grep "^CPU(s)" | tr -dc '0-9') -i /work3/jectli/gapnet/simulations/ -o /work3/jectli/gapnet/spikeogram
```


## `test_model`

````shell
python3 scripts/test_model.py -mf /work3/jectli/gapnet/models -m l3_32_ln512_t1_epoch1000_1 -sd model-at-epoch-40 --use_meta -e moore_et_al_1989 -tf /work3/jectli/gapnet/spikeograms
````
