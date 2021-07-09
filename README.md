# Environment setup

In `conda` (or another virtual environment), do:

```
pip install notebook
pip install tqdm
pip install xgboost
pip install matplotlib
pip install uptools
pip install seutils
```

Then clone this repository.


# Prepare the data

```
python dataset.py signal
python dataset.py bkg
```

This requires the `gfal` command line tools to be installed.


# Train the BDT

See [the notebook](bdt.ipynb).
