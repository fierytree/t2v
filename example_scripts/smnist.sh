export config="smmnist_DDPM_big5"
export data="../data/smnist"

# export exp="../output/ijcai23/smmnist_big5_spynet_cross2"
# export config_mod="training.snapshot_freq=10000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"

# export exp="../output/lvdm/smmnist_64_5c5f5_sqzunet_b2_pmask50_futurepast"
# export exp="../output/lvdm/smmnist_64_5c5f5_v2_pmask50_futurepast"
# export config_mod="data.prob_mask_future=0.5 data.num_frames_future=5 training.snapshot_freq=10000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"


# export exp="../output/lvdm/smmnist_interp_big_c5t5f5_SPADE_lvdm-2"
# export exp="../output/lvdm/smmnist_interp_big_c5t5f5_SPADE_lvdmv2-2"
# export config_mod="model.spade=True data.num_frames_future=5 training.snapshot_freq=10000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"

export exp="../output/ijcai23/smmnist_interp_big_c5t10f5_SPADE_spynet_cross2"
# export exp="../output/lvdm/smmnist_interp_big_c5t10f5_SPADE_lvdm"
# export exp="../output/lvdm/smmnist_interp_big_c5t10f5_SPADE_lvdmv2"
export config_mod="model.spade=True data.num_frames_future=5 data.num_frames=10 training.snapshot_freq=10000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"


# cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
# cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py
CUDA_VISIBLE_DEVICES=7 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}