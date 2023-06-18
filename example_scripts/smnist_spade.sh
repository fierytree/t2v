export config="smmnist_DDPM_big5_spade"
export data="../data/smnist"

export exp="../output/lvdm/smnist_big_c5t5_SPADE"
export config_mod="model.spade=True model.spade_dim=128 training.snapshot_freq=10000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"

CUDA_VISIBLE_DEVICES=7 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}