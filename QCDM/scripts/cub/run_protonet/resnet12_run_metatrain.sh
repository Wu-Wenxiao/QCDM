gpuid=1
set -x
set -e
DATA_ROOT=/home/dancer/filelists/cub/ # path to the json file of CUB
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_pretrain/best_model.tar
MODEL_PATH=./checkpoints/cub/ResNet12_protonet_pretrain-protonet-pretrain-cub-resnet12/best_model.tar
cd ../../../


# echo "============= meta-train 1-shot ============="
# python meta_train.py --extra_dir='-protonet-cub-adapt-v3-resnet12-1shot-adam' --seed 0 --dataset cub --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-3 --epoch 300 --milestones 40 80 --n_shot 1 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
# >log-cub/protonet-cub-adapt-v3-resnet12-1shot-adam.txt
# >log-cub/protonet-cub-resnet12-1shot.txt
echo "============= cub: meta-train 5-shot ============="
python meta_train.py --seed 0 --dataset cub --train_n_way 2 --val_n_way 2 --extra_dir='-protonet-cub-adapt-v3-resnet12-adam-2way-5shot' --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-3 --epoch 300 --milestones 40 80 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
>log-cub/protonet-cub-adapt-v3-resnet12-adam-2way-5shot.txt
# >log-cub/protonet-cub-adapt-v3-resnet12-adam.txt
# >log-cub/protonet-cub-resnet12.txt

