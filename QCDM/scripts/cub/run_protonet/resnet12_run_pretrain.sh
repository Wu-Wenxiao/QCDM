gpuid=1

DATA_ROOT=/home/dancer/filelists/cub/  # path to the json file of CUB
cd ../../../

echo "============= pre-train ============="
python pretrain.py --dataset cub --extra_dir='-protonet-pretrain-cub-resnet12' --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-2 --epoch 150 --milestones 100 --save_freq 10 --val meta --n_shot 1 --val_n_episode 300 \
>log-pretrain/protonet-pretrain-cub-resnet12.txt