gpuid=2
set -x
set -e
DATA_ROOT=/home/dancer/filelists/miniImagenet84/
MODEL_PATH=./pretrain_model/mini2cars/meta_deepbdc/last_model.tar
# MODEL_PATH=./pretrain_model/mini2aircraft/meta_aircraft/last_model.tar
# MODEL_PATH=./pretrain_model/mini2cub/meta_deepbdc/best_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_pretrain/best_model.tar

EXTRA_DIR='-BDC_mini_to_cars-1shot-train'
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-adapt-1euc-layernorm-0.02margin/113.tar
cd ../../../
echo "${gpuid}"
echo "============= meta-train 1-shot ============="
python meta_train_cross.py --mode cross --extra_dir=$EXTRA_DIR --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 500 --milestones 70 --n_shot 1 --train_n_episode 1000 --val_n_episode 600 --reduce_dim 640 --pretrain_path $MODEL_PATH \
>log-cross/BDC_mini_to_cars-1shot-train.txt
# >log-cross/BDC_mini_to_cars-adapt-v3-1shot-train.txt
# >log-cross/BDC_mini_to_cub-1shot.txt
# >log-cross/BDC_mini_to_cars-1shot.txt
# >log-cross/BDC_mini_to_cub-adapt-v3-1shot-onlyregularizer.txt
# >log-cross/BDC_mini_to_cub-adapt-v3-1shot.txt
# >log/BDC-adapt-1euc-adapt-1-shot-product.txt

# echo "============= meta-train 5-shot ============="
# python meta_train_cross.py --mode cross --extra_dir=$EXTRA_DIR --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 500 --milestones 70 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --reduce_dim 640 --pretrain_path $MODEL_PATH \
# >log-cross/BDC_mini_to_cars-train.txt
# >log-cross/BDC_mini_to_cars-adapt-v3-train.txt
# >log-cross/BDC_mini_to_cars-adapt-v3-all.txt
# >log-cross/BDC_mini_to_cars-adapt-v3-base.txt
# >log-cross/BDC_mini_to_aircraft-adapt-v3-onlyregularizer.txt
# >log-cross/BDC_mini_to_aircraft-adapt-v3-regularizerlr0.1.txt
# >log-cross/BDC_mini_to_aircraft-adapt-v3-featurelr0.1-re.txt
# >log-cross/BDC_mini_to_cars-new-adapt-v3.txt
# >log-cross/BDC_mini_to_cars-adapt-v3-seed1.txt
# >log-cross/BDC_mini_to_aircraft-adapt-v3-featurelr0.1.txt
# >log-cross/BDC_mini_to_cars-seed1.txt
# >log-cross/BDC_mini_to_cars-adapt-v3.txt
# >log-cross/BDC_mini_to_cars.txt
# >log-cross/BDC_mini_to_cub-adapt-v3.txt
# >log-cross/BDC_mini_to_aircraft-adapt-v3.txt
# >log-cross/BDC_mini_to_aircraft.txt
# >log-cross/BDC_mini_to_cars_v3_all.txt
# >log-cross-all/BDC_mini_to_cub_v3_all_preall.txt
# >log-cross/BDC_mini_to_cub.txt
# >log-cross/BDC_mini_to_cub_v3_all.txt