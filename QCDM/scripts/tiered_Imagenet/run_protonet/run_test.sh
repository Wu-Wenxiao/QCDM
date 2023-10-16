gpuid=3
# DATA_ROOT=/home/dancer/filelists/tiered-imagenet/specific/
# DATA_ROOT=/home/dancer/filelists/tiered-imagenet224/
DATA_ROOT=/data/tiered_imagenet/

# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_1shot_metatrain/best_model.tar

# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-adapt-1euc-5-shot-v2-layernorm/105.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet/236.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-tiered_imagenet-adapt-v3-layernormlast/39.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-tiered_imagenet-adapt-v3max-layernormlast-weight/453.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrainprotonet-tiered_imagenet-224/81.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet-224-re/20.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet-224-adapt-v3-layernormlast/21.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet-224-adapt-v3-nolayernormlast/14.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet/53.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet-raw/91.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet-raw-adapt-v3-layernormlast/20.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet-raw-adapt-v3-weight/189.tar
# MODEL_1SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_1shot_metatrain-protonet-tiered_imagenet-raw-1shot/88.tar
# MODEL_1SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_1shot_metatrain-protonet-tiered_imagenet-raw-1shot-adapt-v3/49.tar
# MODEL_1SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_1shot_metatrain-protonet-tiered_imagenet-raw-1shot-adapt-v3-weight/164.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain/best_model.tar
MODEL_5SHOT_PATH=/home/dancer/wwx/cam/model/tieredImagenet/protonet/44.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet-adapt-v3-onlyregularizer/best_model.tar
# MODEL_1SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_1shot_metatrain-protonet-tiered_imagenet-raw-1shot-adapt-v3-conv/106.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_5way_5shot_metatrain-protonet-tiered_imagenet-adapt-v3-conv/49.tar
cd ../../../

# N_SHOT=1
# python testv1.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 5 --test_n_episode 2000

N_SHOT=5
python testv1.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_5SHOT_PATH --test_task_nums 5 --test_n_episode 2000