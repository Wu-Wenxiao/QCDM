gpuid=0

set -x
set -e
DATA_ROOT=/home/dancer/filelists/miniImagenet84/
MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_pretrain/best_model.tar

# MODEL_PATH=/home/dancer/wwx/cam/model/miniImagenet/protonet/best_model.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-test/9.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-instance-adapt-no1/187.tar
# MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-instance-adapt-fast/88.tar
# MODEL_PATH=/home/dancer/wwx/DeepBDC-all/checkpoints/mini_imagenet/ResNet12_protonet_5way_5shot_metatrain-5queryproto-sup-0.5-adapt-0.1/last_model.tar
cd ../../../


# echo "============= meta-train 1-shot ============="
# python meta_train.py --extra_dir='-protonet-mini-10way-1shot' --train_n_way 10 --val_n_way 10 --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 200 --milestones 40 80 --n_shot 1 --train_n_episode 1000 --val_n_episode 600 --pretrain_path $MODEL_PATH\
# >log-10/protonet-mini-10way-1shot.txt
# >log-10/protonet-mini-adapt-v3-10way-1shot.txt

echo "============= meta-train 5-shot ============="
python meta_train.py --extra_dir='-protonet-mini-adapt-v3-5way-5-shot-weight-simple' --train_n_way 5 --val_n_way 5 --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 300 --milestones 40 80 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
>log-10/protonet-mini-adapt-v3-5way-5-shot-weight-simple.txt
# >log-10/protonet-mini-adapt-v3-5way-5-shot-simple.txt
# >log-10/protonet-mini-adapt-v3-5way-featurefix.txt
# >log-10/protonet-mini-adapt-v3-2-way.txt
# >log-10/protonet-mini-adapt-v3-euc-se.txt
# >log-10/protonet-mini-euc-se.txt
# >log-10/protonet-mini-adapt-v3-cos-re.txt
# >log-test/protonet-mini-5way-5shot-2MLP-weight.txt
# >log-test/protonet-mini-5way-5shot-2MLP.txt
# >log-test/protonet-mini-3way-5shot.txt
# >log-10/protonet-mini-adapt-v3-cos-weight.txt
# >log-10/protonet-mini-adapt-v3-cos.txt


# >log-10/protonet-mini-cos.txt
# >log-10/protonet-mini-adapt-v3-10way-5shot.txt
# >log-10/protonet-mini-10way-5shot.txt

# echo "============= meta-train 10-way 10-shot ============="
# python meta_train.py --extra_dir='-protonet-mini-adapt-v3-10way-10shot' --train_n_way 10 --val_n_way 10 --seed 0 --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 300 --milestones 40 80 --n_shot 10 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
# >log-10/protonet-mini-adapt-v3-10way-10shot.txt
# >log-10/protonet-mini-10way-10shot.txt


# >log/protonet-mini-adapt-var-1shot.txt
# >log/protonet-mini-adapt-v3-1shot-weight.txt
# >log-conv/miniImagenet/protonet-mini-adapt-v3-conv.txt
# >log/protonet-adapt-1euc-layernorm-1-shot.txt

# echo "============= meta-train 10-shot ============="
# python meta_train.py --seed 0 --dataset mini_imagenet --extra_dir='-proto-mini-adapt-v3-10shot' --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 500 --milestones 40 80 --n_shot 10 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH \
# >log-10/proto-mini-adapt-v3-10shot.txt
# >log-10/proto-mini-10shot.txt
# >log/protonet-mini-adapt-v3-onlyregularizer.txt
# >log/protonet-mini-adapt-v3-nolayernorm-onlyregularizer-pre.txt
# >log/protonet-mini-adapt-v3-nolayernorm-onlyregularizer.txt
# >log/protonet-mini-adapt-var-nolayernorm.txt
# >log/protonet-mini-adapt-anova.txt
# >log/protonet-mini-adapt-var.txt
# >log/protonet-mini-adapt-v3-weight.txt
# >log/protonet-mini-adapt-v3.txt
# >log/protonet-adapt-v3-layernormlast-weight.txt
# >log/protonet-adapt-v3-layernormlast-nosquare.txt
# > log/protonet-instance-adapt-no1-channel-adapt-v11.txt
# > log/protonet-instance-adapt-no1-channel-adapt-parttrain.txt
# > log/protonet-instance-adapt-no1-channel-adapt.txt
# > log/protonet-instance-adapt-no1.txt
# > log/protonet-instance-adapt-fast.txt
# > log/protonet-instance-adapt.txt
# > log/protonet-layernorm.txt
# > log/protonet-adapt-standard-first-and-last.txt
# >log/protonet-adapt-0.03margin-yc.txt
# >log/protonet-adapt-layernorm-0.001margin-yc.txt
# >log/protonet-adapt-layernorm-0.001margin.txt
# >log/protonet-adapt-1euc-1lossu.txt
# >log/protonet-adapt-1euc-Square.tx
# >log/protonet-adapt-1euc-layernorm+proto.txt
# >log/protonet-adapt-1euc-layernorm-0.02margin.txt
# >log/protonet-adapt-1euc-layernorm.txt
# >log/protonet-margin-0.03v2-adapt-1euc.txt
# >log/protonet-margin-0.1v2-adapt-0.1-2lossu.txt
# >log/protonet-margin-0.5-adapt-0.1.txt
# >log/protonet-5queryproto-sup-0.5-adapt-0.1.txt
# >log/protonet-5queryproto-sup-2.0.txt
# >log/protonet-5queryproto-sup-1.0.txt
# >log/protonet-5queryproto-sup-0.5.txt
# >log/protonet-query-sup-0.2.txt