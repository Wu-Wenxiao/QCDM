gpuid=0
set -x
set -e

DATA_ROOT=/home/dancer/filelists/miniImagenet84/
# MODEL_PATH=./checkpoints/cross_miniImagenet/protonet/100.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_pretrain/best_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_pretrainmini_to_cub_all_pretrain/best_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-test/9.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-instance-adapt-no1/187.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-instance-adapt-fast/88.tar
# MODEL_PATH=/home/dancer/wwx/DeepBDC-all/checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-5queryproto-sup-0.5-adapt-0.1/last_model.tar
MODEL_PATH=./pretrain_model/mini2cub/protonet/best_model.tar
# MODEL_PATH=./checkpoints/pretrain-model/mini2cars-wwx/protonet/last_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_pretrain/best_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_pretrain/best_model.tar
# MODEL_PATH=./checkpoints/pretrain-model/mini2cars-wwx/protonet/100.tar
cd ../../../


# echo "============= meta-train 1-shot ============="
# python meta_train_cross.py --mode cross --extra_dir="-protonet-mini2cars-adapt-v3-1shot-all" --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 1000 --milestones 800 --n_shot 1 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH\
# >log-cross/protonet-mini2cars-adapt-v3-1shot-all.txt
# >log-cross/protonet-mini2cars-1shot-all.txt
# >log-cross/protonet-mini2cub-adapt-v3-1shot-all.txt
# >log-cross/protonet-mini2cub-adapt-v3-1shot-base.txt # pretrain为base上训练的
# >log/protonet-adapt-1euc-layernorm-1-shot.txt
echo "${gpuid}"
echo "============= meta-train 5-shot ============="
python meta_train_cross.py --train_n_way 2 --val_n_way 2 --mode cross --extra_dir="-mini_to_cub-protonet-adapt-v3-2way-5shot" --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 300 --milestones 70 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
>log-cross/mini_to_cub-protonet-adapt-v3-2way-5shot.txt
# python meta_train_cross.py --mode cross --extra_dir="-protonet-mini2cub-1shot" --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 500 --milestones 200 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
# >log-cross/mini_to_cars-protonet-wwx-adapt-v3-base.txt
# >log-cross/mini_to_cars-protonet-wwx-adapt-v3-100.txt
# >log-cross/mini_to_cars-protonet-wwx-adapt-v3.txt
# >log-cross/mini_to_cars-protonet-wwx.txt
# >log-cross/mini_to_aircraft-protonet-wwx.txt
# >log-cross/mini_to_cub-protonet-adapt-v3.txt
# >log-cross/mini_to_aircraft-protonet.txt
# >log-cross/mini_to_aircraft_v3_layernormlast_all.txt
# >log-cross/mini_to_cars_v3_layernormlast_all-weight.txt
# >log-cross/mini_to_cars_v3_layernormlast_all.txt
# >log-cross-all/mini_to_cub_v3_layernorm_all_preall-best.txt
# >log-cross-all/mini_to_cub_v3_layernorm_all_preall.txt
# >log-cross/mini_to_cub_v3_layernorm_all.txt