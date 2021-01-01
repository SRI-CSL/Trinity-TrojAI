# Trinity-TrojAI
This repository contains code developed by the SRI team for the IARPA/TrojAI program (https://pages.nist.gov/trojai/) .  

## Relevant Publications from SRI on Trojans in AI

* Karan Sikka, Indranil Sur, Susmit Jha, Anirban Roy, and Ajay Divakaran. "Detecting Trojaned DNNs Using Counterfactual Attributions." arXiv preprint arXiv:2012.02275 (2020).

* Panagiota Kiourti, Kacper Wardega, Jha, Susmit, & Li, Wenchao. TrojDRL: Trojan Attacks on Deep Reinforcement Learning Agents. ArXiv, abs/1903.06638 (2019).

## Overall Approach

Given a model and some clean images (without a Trojan), we extract features that describe that model. These features are then used by a classifier to discriminate between Trojaned and non-Trojaned models. We have two different pipelines for learning the features:

* Attribution-based approach

* Trigger-search through input editing

## How to run the code

We assume the metadata is in `data/round3-dataset-train/METADATA.csv` and that the models are in `data/round3-dataset-train/models/id-0000xxxx`.

### Attribution-based Trojan Detection 

- Setup local paths: Set the troj_root in utils_nn.py and the local logging directory in run_trojan_detector.py of the AttributionBased folder.
- Run the following feature extractor (needs to be run twice for each model)

```
python get_attribution_features_trojai.py --model_name model_name --model_dir model_src --meta_data meta_data_path --attributions_dir path_attribution_features --results_dir path_curve_features --attribution_fn ATTRIB_FN --round ROUND # example run of feature extractor for attribution-based Trojan detection, set the paths to model directory, metafile; the output locations; and the choice of attribution function along with the round ID for the TrojAI datset. 
```

- Run the following for classifying the test models to be Trojaned or not Trojaned.

```
python run_trojan_detector.py --test_split 10 --val_split 10 --nfolds 5 --arch conv --epochs 100 --lr 0.01 --p_tx 0.8 --nhead 2 --nlayers_tx 2 --dataset mnist --patience 20 --num_ensemble 5 --nhid 16  # example detector run with suggested hyperparams
```

### Feature Extraction for Trigger-Generation

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





#### Contact: susmit.jha@sri.com 
