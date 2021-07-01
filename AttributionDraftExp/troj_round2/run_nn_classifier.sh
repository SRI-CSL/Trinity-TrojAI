SPLIT=10
LR=1e-3
DROPOUT_TX=0.8
NHEAD=2
NLAYERS_TX=2
EPOCHS=2
ARCH="conv"
# OUTPUT="outputs/transformer_lr:1e-4.txt
OUTPUT="tmp_runs/tx_dropout0.8Pos_4head.txt"
CUDA_VISIBLE_DEVICES=0

python run_nn_classifier_troj.py --test_split=$SPLIT --val_split=$SPLIT --nfolds=5 --arch=conv --output_file $OUTPUT --arch $ARCH --epochs $EPOCHS --lr $LR --p_tx $DROPOUT_TX --nhead $NHEAD --nlayers_tx $NLAYERS_TX --use_wandb
