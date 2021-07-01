ARCH="transformer"
OUTPUT="outputs/transformer_20.txt"

python run_nn_classifier_mnist.py --test_split=20 --val_split=20 --nfolds=5 --arch=conv --dump_output --output_file $OUTPUT --test_split 20 --val_split 20 --arch $ARCH --epochs 100