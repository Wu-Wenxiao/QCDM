gpuid=2
set -x
set -e
DATA_ROOT=/data/tiered_imagenet/
MODEL_PATH=./pretrain_model/tieredImagenet/meta_deepbdc/best_model.tar
cd ../../../
echo "${gpuid}"
echo "============= meta-train 1-shot ============="
python meta_train_bdc.py --seed 0 --extra_dir='-BDC-tiered-1shot-adapt-v3' --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 300 --milestones 70 --n_shot 1 --train_n_episode 1000 --val_n_episode 1000 --reduce_dim 256 --pretrain_path $MODEL_PATH \
>log-tiered/BDC-tiered-1shot-adapt-v3.txt
# >log-tiered/BDC-tiered-1shot.txt

# echo "============= meta-train 5-shot ============="
# python meta_train_bdc.py --seed 0 --extra_dir='-BDC-tiered-adapt-v3-sigmoid' --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 300 --milestones 70 --n_shot 5 --train_n_episode 1000 --val_n_episode 1000 --reduce_dim 256 --pretrain_path $MODEL_PATH \
# >log-tiered/BDC-tiered-adapt-v3-sigmoid.txt
# >log-tiered/BDC-tiered-adapt-v3-weight.txt
# >log-tiered/BDC-tiered-adapt-v3.txt
# >log-tiered/BDC-tiered.txt