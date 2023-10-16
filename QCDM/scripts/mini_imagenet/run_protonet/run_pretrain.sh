gpuid=1

DATA_ROOT=/home/dancer/filelists/miniImagenet84/
cd ../../../

echo "============= pre-train ============="
python pretrain_cross.py --seed 0 --extra_dir='-mini_to_cars_all_pretrain' --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-2 --wd 1e-4 --epoch 200 --milestones 100 150 --save_freq 100 --val meta --val_n_episode 600 --n_shot 5 \
# >log-pretrain/mini_to_cars_all_pretrain.txt
# >log-pretrain/mini_to_cub_all_pretrain.txt