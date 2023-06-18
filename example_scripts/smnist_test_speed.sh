export config="smmnist_DDPM_big5"
export data="../data/smnist"

# export exp="../output/lvdm/smmnist_big5_sqz8_sf_split"
# export exp="../output/lvdm/smmnist_big5_sqzunet"
# export exp="../output/smmnist_big5-snapshot_1000"
# export config_mod="training.snapshot_freq=1000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"

# export exp2="../output/lvdm/smmnist_big5_sqzunet-snapshot_1000"
# export config_mod="training.snapshot_freq=1000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"

# export exp="../output/lvdm/smmnist_interp_big_c5t5f5_SPADE_lvdm"
export exp="../output/lvdm/smmnist_interp_big_c5t5f5_SPADE_lvdmv2"
export config_mod="data.num_frames_future=5 training.snapshot_freq=10000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"


cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=5 python main.py --test_speed --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
