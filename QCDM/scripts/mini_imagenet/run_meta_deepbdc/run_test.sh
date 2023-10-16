gpuid=3
cd ../../../

DATA_ROOT=/home/dancer/filelists/miniImagenet84/
# DATA_ROOT=/home/dancer/filelists/cub/ 
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain/best_model.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-mini-1shot/55.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain/best_model.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-mini/65.tar
MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-adapt-1euc-layernorm-0.02margin/113.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC_mini_to_cub_v3_all/59.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-adapt-1euc-layernorm/43.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-adapt-1euc-adapt-1-shot-product/100.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-adapt-v3-weight-next/96.tar
# MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-mini/65.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-mini-adapt-v3-1shot-weight/36.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-mini-1shot/55.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-mini-adapt-1shot-euc/235.tar
# MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-mini-adapt-1shot-cos/281.tar

# MODEL_10SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_10shot_metatrain-BDC-mini-10shot/96.tar

# N_SHOT=1
# python testv1.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 5 --test_n_episode 2000

N_SHOT=5
python testv1.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_5SHOT_PATH --test_task_nums 5 --test_n_episode 2000


# N_SHOT=10
# python testv1.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_10SHOT_PATH --test_task_nums 5 --test_n_episode 2000