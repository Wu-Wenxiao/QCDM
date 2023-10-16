gpuid=3
N_SHOT=5

DATA_ROOT=/home/dancer/filelists/cub/  # path to the json file of CUB
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_${N_SHOT}shot_metatrain/best_model.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_5shot_metatrain-adapt-5shot-v3/278.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_5shot_metatrain-adapt-5shot-v3-layernormlast/178.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_5shot_metatrain-5shot/45.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_5shot_metatrain-adapt-5shot-v3max-layernormlast-weight/235.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_1shot_metatrain-protonet-cub/84.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_1shot_metatrainprotonet-cub-1-shot-adapt-v3-noaug/77.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_1shot_metatrainprotonet-cub-1-shot-adapt-v3-noaug-sigmoid/85.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_1shot_metatrain-protonet-cub-1-shot-adapt-v3-conv/74.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_5shot_metatrain-protonet-cub-adapt-v3-adam/47.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_5shot_metatrain-protonet-cub-adapt-v3-adam-weight-crop/42.tar
# MODEL_PATH=./checkpoints/cub/ResNet12_protonet_5way_5shot_metatrain-protonet-cub--resnet12/85.tar
# MODEL_PATH=./checkpoints/cub/ResNet12_protonet_5way_5shot_metatrain-protonet-cub-adapt-v3-resnet12-adam/49.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_1shot_metatrain-protonet-cub-1shot-adapt-v3-adam/46.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_protonet_5way_1shot_metatrain-protonet-cub-1-shot/84.tar
# MODEL_PATH=./checkpoints/cub/ResNet12_protonet_5way_1shot_metatrain-protonet-cub-resnet12-1shot/48.tar
# MODEL_PATH=./checkpoints/cub/ResNet12_protonet_5way_1shot_metatrain-protonet-cub-adapt-v3-resnet12-1shot-adam/77.tar
MODEL_PATH=./checkpoints/cub/ResNet12_protonet_5way_5shot_metatrain-protonet-cub--resnet12/85.tar
# MODEL_PATH=./checkpoints/cub/ResNet12_protonet_2way_5shot_metatrain-protonet-cub-adapt-v3-resnet12-adam-2way-5shot/97.tar

cd ../../../

# python testv1.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method protonet --image_size 224 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 512 --model_path $MODEL_PATH --test_task_nums 5
python testv1.py --test_n_way 5 --dataset cub --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 512 --model_path $MODEL_PATH --test_task_nums 5