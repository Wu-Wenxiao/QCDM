gpuid=0
set -x
set -e
# DATA_ROOT=/home/dancer/filelists/tiered-imagenet/specific/
# DATA_ROOT=/home/dancer/filelists/tiered-imagenet224/
DATA_ROOT=/data/tiered_imagenet/

# MODEL_PATH=./checkpoints/pretrain-model/tiered_Imagenet/protonet-resnet12/best_model.tar # 84*84
# MODEL_PATH=./checkpoints/tiered_imagenet/ResNet12_protonet_pretrainprotonet-tiered_imagenet-pretrain-224-next/best_model.tar # 224*224
# MODEL_PATH=./pretrain_model/tieredImagenet/protonet/best_model.tar

MODEL_PATH=/home/dancer/wwx/cam/model/tieredImagenet/protonet/44.tar
cd ../../../

echo "${gpuid} "
echo "============= tiered_imagenet================"
# echo "============= meta-train 1-shot ============="
# python meta_train.py --seed 0 --extra_dir="-protonet-tiered_imagenet-raw-1shot-adapt-v3-conv" --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 300 --milestones 70 --n_shot 1 --train_n_episode 1000 --val_n_episode 1000 --pretrain_path $MODEL_PATH \
# >log-conv/tieredImagenet/protonet-tiered_imagenet-raw-1shot-adapt-v3-conv.txt
# >log-tiered/protonet-tiered_imagenet-raw-1shot-adapt-v3-weight.txt
# >log-tiered/protonet-tiered_imagenet-raw-1shot-adapt-v3.txt
# >log-tiered/protonet-tiered_imagenet-raw-1shot.txt

echo "============= meta-train 5-shot ============="
python meta_train.py --seed 0 --extra_dir="-protonet-tiered_imagenet-adapt-v3-conv" --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 300 --milestones 70 --n_shot 5 --train_n_episode 1000 --val_n_episode 1000 --pretrain_path $MODEL_PATH \
>log-tiered/protonet-tiered_imagenet-adapt-v3-conv.txt
# >log-tiered/protonet-tiered_imagenet-adapt-v3-onlyregularizer.txt
# >log-tiered/protonet-tiered_imagenet-raw-adapt-v3-weight.txt
# >log-tiered/protonet-tiered_imagenet-raw-seed1.txt
# >log-tiered/protonet-tiered_imagenet-raw-adapt-v3-layernormlast.txt
# >log-tiered/protonet-tiered_imagenet-raw.txt

# python meta_train.py --seed 0 --extra_dir="-protonet-tiered_imagenet-raw" --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 100 --milestones 70 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
# >log-tiered/protonet-tiered_imagenet-224-adapt-v3-nolayernormlast.txt
# >log-tiered/protonet-tiered_imagenet-224-adapt-v3-layernormlast.txt
# >log-tiered/protonet-tiered_imagenet-224.txt
# > log-tiered/protonet-tiered_imagenet-adapt-v3max-layernormlast-weight.txt
# > log-tiered/protonet-tiered_imagenet-adapt-v3-nolayernorm.txt
# > log-tiered/protonet-tiered_imagenet-adapt-v3-layernormlast.txt
# > log-tiered/protonet-tiered_imagenet.txt