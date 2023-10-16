gpuid=1
set -x
set -e

DATA_ROOT=/home/dancer/filelists/cub/ # path to the json file of CUB
MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_pretrain/last_model.tar
cd ../../../

# echo "============= meta-train 1-shot ============="
# python meta_train_bdc.py --seed 0 --dataset cub --extra_dir='BDC-cub-1-shot-adapt-v3-noaug-weight' --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 200 --milestones 40 --n_shot 1 --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --pretrain_path $MODEL_PATH \
# >log-cub/BDC-cub-1-shot-adapt-v3-noaug-weight.txt
# >log-cub/BDC-cub-1-shot-adapt-v3-noaug.txt
# >log-cub/BDC-cub-1-shot.txt
echo "============= meta-train 5-shot ============="
# python meta_train_bdc.py --seed 0 --dataset cub --extra_dir='-BDC-cub-adapt-v3-adam-1e-4' --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --lr 1e-4 --epoch 300 --milestones 150 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --pretrain_path $MODEL_PATH \
python meta_train_bdc.py --seed 0 --dataset cub --extra_dir='-BDC-cub-adapt-v3' --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 500 --milestones 40 200 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --reduce_dim 256 --pretrain_path $MODEL_PATH \
>log-cub/BDC-cub-adapt-v3.txt
# >log-cub/BDC-cub.txt
# >log-cub/BDC-cub-adapt-v3-adam-3-4.txt
# >log-cub/BDC-cub-adapt-v3-adam-1e-4.txt
# >log-cub/BDC-cub-adapt-v3-adam.txt
# >log-cub/BDC-adapt-1euc-cub-weight-re.txt
# >log-cub/BDC-adapt-1euc-cub-weight.txt
# >log-cub/BDC-adapt-1euc-cub.txt