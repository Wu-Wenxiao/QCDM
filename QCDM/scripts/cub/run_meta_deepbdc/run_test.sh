gpuid=3
N_SHOT=1

DATA_ROOT=/home/dancer/filelists/cub/ # path to the json file of CUB
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_5shot_metatrain-adapt-1euc-cub/49.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_${N_SHOT}shot_metatrain/best_model.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_5shot_metatrain-adapt-1euc-cub/75.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_5shot_metatrain-adapt-1euc-cub-weight-re/183.tar
# MODEL_PATH=./pretrain_model/mini2cub/meta_deepbdc/best_model.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_pretrain/last_model.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_1shot_metatrainBDC-cub-1-shot/33.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_1shot_metatrainBDC-cub-1-shot-adapt-v3-noaug/23.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_1shot_metatrainBDC-cub-1-shot-adapt-v3-noaug-weight/61.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_1shot_metatrainBDC-cub-1-shot-adapt-v3-noaug-weight/28.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_5shot_metatrain-BDC-cub-adapt-v3-adam-3-4/6.tar
# MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_5way_5shot_metatrain-BDC-cub/40.tar
# MODEL_PATH=./checkpoints/cub/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-cub-resnet12/55.tar
# MODEL_PATH=./checkpoints/cub/ResNet12_meta_deepbdc_5way_5shot_metatrain-BDC-cub-adapt-v3-resnet12/92.tar
# MODEL_PATH=./checkpoints/cub/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-cub-resnet12-1shot/48.tar
MODEL_PATH=./checkpoints/cub/ResNet12_meta_deepbdc_5way_1shot_metatrain-BDC-cub-adapt-v3-resnet12-1shot/125.tar
cd ../../../

# python testv1.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 --test_n_episode 2000
python testv1.py --dataset cub --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_PATH --test_task_nums 5 --test_n_episode 2000