gpuid=3
# DATA_ROOT=/home/dancer/filelists/miniImagenet84/
DATA_ROOT=/home/dancer/filelists/cub/
# DATA_ROOT=/home/dancer/filelists/car_all/
# DATA_ROOT=/home/dancer/filelists/Aircraft_fewshot/
# DATA_ROOT=/home/dancer/filelists/car_all/cars_for_fsl_split/

# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_1shot_metatrain-adapt-1euc-layernorm-1-shot/124.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-adapt-1euc-layernorm-1-shot/21.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cub_v3_layernorm_all/360.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cub_v3_layernorm_all_preall/26.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cars_v3_layernormlast_all/264.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cars_v3_layernormlast_all-weight/41.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cars_v3_layernormlast_all-weight/32.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrainmini_to_aircraft_v3_layernormlast_all/409.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cars_v3_layernormlast_all/111.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrainmini_to_aircraft-protonet/78.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_pretrainmini_to_cub_all_pretrain/best_model.tar
MODEL_5SHOT_PATH=./pretrain_model/mini2cub/protonet/best_model.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrainmini_to_cub-protonet-adapt-v3/18.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_aircraft-protonet-wwx/9.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cars-protonet-wwx/11.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cars-protonet-wwx-adapt-v3/54.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrainmini_to_aircraft_v3_layernormlast_all/165.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cars-protonet-wwx-adapt-v3-base/476.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-mini_to_cars_v3_layernormlast_all/111.tar

# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_1shot_metatrain-protonet-mini2cub-adapt-v3-1shot-base/659.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_1shot_metatrain-protonet-mini2cars-adapt-v3-1shot-all/8.tar

# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_1shot_metatrain-protonet-mini2ars-1shot-all/171.tar

cd ../../../

# N_SHOT=1
# python testv1.py --dataset cub --test_n_way 2 --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 5 --test_n_episode 2000

N_SHOT=5
python testv1.py --dataset cub --test_n_way 5 --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_5SHOT_PATH --test_task_nums 5 --test_n_episode 2000