gpuid=0

DATA_ROOT=/path/mini_imagenet
cd ../../../

echo "============= pre-train ============="
python pretrain.py --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 170 --milestones 100 150 --save_freq 100 --reduce_dim 640 --dropout_rate 0.8 --val meta --val_n_episode 600