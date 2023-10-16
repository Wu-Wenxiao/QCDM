gpuid=0
set -x
set -e
DATA_ROOT=/home/dancer/filelists/cub/ # path to the json file of CUB
MODEL_PATH=./checkpoints/cub/ResNet18_protonet_pretrain/best_model.tar
cd ../../../


# echo "============= meta-train 1-shot ============="
# python meta_train.py --extra_dir='-protonet-cub-1shot-adapt-v3-adam-weight' --seed 0 --dataset cub --data_path $DATA_ROOT --model ResNet18 --method protonet --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 300 --milestones 40 80 --n_shot 1 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
# >log-cub/protonet-cub-1shot-adapt-v3-adam-weight.txt
# >log-cub/protonet-cub-1shot-adapt-v3-adam.txt
# >log-cub/protonet-cub-1-shot-adapt-v3-noaug-featurefix.txt
# >log-conv/cub/protonet-cub-1-shot-adapt-v3-conv.txt
# >log-cub/protonet-cub-1-shot-adapt-v3-noaug-sigmoid.txt
# >log-cub/protonet-cub-1-shot-adapt-v3-noaug-nolayernorm-sigmoid.txt
# >log-cub/protonet-cub-1-shot-adapt-v3-noaug-nolayernorm.txt
# >log-cub/protonet-cub-1-shot-adapt-v3-noaug.txt
# >log-cub/protonet-cub-1-shot.txt
echo "============= cub: meta-train 5-shot ============="
python meta_train.py --seed 0 --train_n_way 2 --val_n_way 2 --dataset cub --extra_dir='-protonet-cub-adapt-v3-adam-2way-5shot' --data_path $DATA_ROOT --model ResNet18 --method protonet --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 500 --milestones 40 80 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
>log-cub/protonet-cub-adapt-v3-adam-2way-5shot.txt
# >log-cub/protonet-cub-adapt-v3-adam-weight-crop.txt
# >log-cub/protonet-cub-adapt-v3-adam-weight.txt
# >log-cub/protonet-cub-adapt-v3-adam.txt
# >log-cub/protonet-cub-adapt-v3-weight.txt
# >log-cub/protonet-cub-adapt-v3.txt
# >log-cub/protonet-adapt-5shot-v3max-layernormlast-weight.txt
# >log-cub/protonet-adapt-5shot-v3-0.001.txt
# >log-cub/protonet-5shot.txt
# >log-cub/protonet-adapt-5shot-v3-layernormlast.txt
# >log-cub/protonet-adapt-5shot-v3.txt