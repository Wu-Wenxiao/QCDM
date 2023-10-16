gpuid=0
set -xe
DATA_ROOT=/home/dancer/filelists/tiered-imagenet224/
cd ../../../

echo "============= pre-train ============="
# python pretrain.py --seed 0 --extra_dir=protonet-tiered_imagenet-pretrain-224 --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-2 --wd 1e-4 --epoch 200 --milestones 100 150 --save_freq 10 --val meta --val_n_episode 600 --n_shot 5 \
python pretrain.py --seed 0 --extra_dir=protonet-tiered_imagenet-pretrain-224-next --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-3 --wd 1e-4 --epoch 80 --milestones 30 --save_freq 10 --val meta --val_n_episode 600 --n_shot 5 \
>log-tiered/protonet-tiered_imagenet-pretrain-224-next.txt
# >log-tiered/protonet-tiered_imagenet-pretrain-224.txt