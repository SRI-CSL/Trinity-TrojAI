Reverse engineering features are extracted using three different versions of
reverse engineering, corresponding to the `extract_fv_v17.py`, `extract_fv_color_v2r.py`,
and `extract_fv_color_v2xy.py` files. The `extract_fv_v17.py` feature extractor
tries to find polygons that trigger the model while the `extract_fv_color_v2r.py`
and `extract_fv_color_v2xy.py` extractors try to reverse engineer color filters
that trigger the model. Each feature extractor extracts raw information
that needs to be further processed in the next step before being fed through the
trojan classifier.

In the second stage the features need to be pre-processed and then fed
through the trojan clasifier. Both the trojan classifier itself and these
pre-processing steps are implemented in the forward
function at line 82 of `mlp_set_color_v2xy.py`.

The `demo.py` script combines these together to predict the probability a
specific model has a trojan in it.

## Common Tasks

Here we walk through some supported functionality for round 3 of the trojai dataset.
We assume the round 3 metadata is in `data/round3-dataset-train/METADATA.csv`
and that the models are in `data/round3-dataset-train/models/id-0000xxxx`.

### Feature Extraction

To extract features for an entire round of the TrojAI dataset run all three
feature extraction scripts:

```
$ python extract_fv_v17r.py       # for polygon reverse engineering
$ python extract_fv_color_v2r.py  # for color filter reverse engineering
$ python extract_fv_color_v2xy.py # for color filter reverse engineering with position embedding
```

This creates feature files like `fvs_polygon_r3_v2.pt` containing features for
all the models. Use the `--start` and `--end` parameters to specify a range
of models to extract features from.

Next, parse the features into the `data.pt` file consumed by the other steps.

```
$ python parse_r3v3r.py
```

### Training the Trojan Classifier

To train the trojan classifier we perform a hyperparameter search using
8-fold cross validation.
__This requires hyperopt (`pip install hyperopt`).__
It assumes the `data.pt` file from the previous step has already been
generated and can be run with

```
$ python crossval_hyper.py
```

The hyperparameter search never ends, but the best model will be kept under `sessions/xxxxxxx/model.pt`.
Performance can be read from `sessions/xxxxxxx/log.txt`.
See `demo.py` for an example of how to use that model file.


### Building a Singularity Image for Submission to the TrojAI Leaderboard

Once the model is trained it's easy to submit it to the leaderboard.
Edit the top of `demo.py` to use the appropriate checkpoint generated from
the hyperparameter search then build the singularity image.

```
sudo singularity build trojai.simg real_trojan_detector.def
```


