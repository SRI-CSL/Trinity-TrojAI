## Attribution Based Detection

- Setup local paths: Set the troj_root in utils_nn.py and the local logging directory in run_trojan_detector.py
- Run the following feature extractor (needs to be run twice for each model)

python get_attribution_features_trojai.py --model_name model_name --model_dir model_src --meta_data meta_data_path --attributions_dir path_attribution_features --results_dir path_curve_features --attribution_fn ATTRIB_FN --round ROUND

- Run the following for classifying the test models to be Trojaned or not Trojaned.

python run_trojan_detector.py --test_split 10 --val_split 10 --nfolds 5 --arch conv --epochs 100 --lr 0.01 --p_tx 0.8 --nhead 2 --nlayers_tx 2 --dataset mnist --patience 20 --num_ensemble 5 --nhid 16
