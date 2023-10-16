gpuid=0
cd ../../../

DATA_ROOT=/data/tiered_imagenet/
# MODEL_1SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain/best_model.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain/best_model.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-tiered/84.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-tiered-adapt-v3/158.tar
# MODEL_5SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-tiered-adapt-v3-weight/63.tar
MODEL_1SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-tiered-1shot/84.tar
MODEL_1SHOT_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-tiered-1shot-adapt-v3/70.tar
N_SHOT=1
python testv1.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_1SHOT_PATH --test_task_nums 5 --test_n_episode 2000

# N_SHOT=5
# python testv1.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_5SHOT_PATH --test_task_nums 5 --test_n_episode 2000
