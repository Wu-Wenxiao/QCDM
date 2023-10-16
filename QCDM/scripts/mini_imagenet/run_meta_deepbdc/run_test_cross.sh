gpuid=3
set -x
set -e
# DATA_ROOT=/home/dancer/filelists/miniImagenet84/
# DATA_ROOT=/home/dancer/filelists/cub/
# DATA_ROOT=/home/dancer/filelists/Aircraft_fewshot/
# DATA_ROOT=/home/dancer/filelists/car_all/
DATA_ROOT=/home/dancer/filelists/cars/
# DATA_ROOT=/home/dancer/filelists/car_all/cars_for_fsl_split/

# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_1shot_metatrain-adapt-1euc-layernorm-1-shot/124.tar

# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cub/74.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cub_v3_all/367.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars_v3_all/192.tar
# MODEL_1SHOT_PATH=./pretrain_model/mini2cub/meta_deepbdc/best_model.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC_mini_to_cub-adapt-v3-1shot/15.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars/10.tar
# MODEL_1SHOT_PATH=./pretrain_model/mini2cub/meta_deepbdc/best_model.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_aircraft/best_model.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cub-adapt-v3/94.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_aircraft-adapt-v3/99.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars/43.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars-adapt-v3/84.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_aircraft-adapt-v3-featurelr0.1/80.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars-seed1/71.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars-adapt-v3-seed1/59.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_aircraft-adapt-v3-featurelr0.1/89.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars-new/93.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars-new-adapt-v3/45.tar
# MODEL_5SHOT_PATH=/home/dancer/wwx/DeepBDC-all/checkpoints/pretrain-model/best_model.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_aircraft-adapt-v3-regularizerlr0.1/41.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars-adapt-v3-all/189.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC_mini_to_cub-adapt-v3-1shot/15.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC_mini_to_cub-adapt-v3-1shot-onlyregularizer/40.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC_mini_to_cars-1shot/10.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC_mini_to_cars-adapt-v3-1shot/163.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC_mini_to_cub-1shot/64.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC_mini_to_cars-adapt-v3-1shot-train/59.tar
MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars-adapt-v3-train/59.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC_mini_to_cars-1shot-train/4.tar

# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cars-train/9.tar
cd ../../../

# N_SHOT=1
# python testv1.py --dataset cars --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 5 --test_n_episode 2000

N_SHOT=5
python testv1.py --dataset cars --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_5SHOT_PATH --test_task_nums 5 --test_n_episode 2000