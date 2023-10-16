gpuid=0
set -x
set -e
DATA_ROOT=/home/dancer/filelists/miniImagenet84/
MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_pretrain/best_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-adapt-v3-weight/last_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-adapt-1euc-layernorm-0.02margin/113.tar
cd ../../../

echo "${gpuid}"
# echo "============= meta-train 1-shot ============="
# python meta_train_bdc.py --extra_dir='-BDC-mini-adapt-1shot-euc' --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 500 --milestones 40 80 --n_shot 1 --train_n_episode 1000 --val_n_episode 600 --reduce_dim 640 --pretrain_path $MODEL_PATH \
# >log/BDC-mini-adapt-1shot-euc.txt
# >log/BDC-mini-adapt-1shot.txt
# >log/BDC-mini-adapt-v3-1shot-weight.txt
# >log/BDC-adapt-1euc-adapt-1-shot-product.txt

echo "============= meta-train 10-shot ============="
python meta_train_bdc.py --seed 0 --extra_dir='-BDC-mini-adapt-v3-10shot' --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 500 --milestones 40 80 --n_shot 10 --train_n_episode 600 --val_n_episode 600 --reduce_dim 640 --pretrain_path $MODEL_PATH \
>log-10/BDC-mini-adapt-v3-10shot.txt
# >log-10/BDC-mini-10shot.txt
# >log/BDC-mini.txt
# >log/BDC-adapt-v3-weight-next.txt
# >log/BDC-adapt-v3-weight.txt
# >log/BDC-adapt-1euc-adapt-instance.txt
# >log/BDC-adapt-1euc-re.txt
# >log/BDCt-adapt-1euc.txt