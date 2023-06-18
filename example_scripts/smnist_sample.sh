export config="smmnist_DDPM_big5"
export data="../data/smnist"
export nfp="10"

# export exp="../output/lvdm/smmnist_big5_sqzunet"
# export exp=../output/lvdm/smmnist_big5_sqz8_sf_split
# export exp=../output/lvdm/smmnist_64_5c5f5_sqzunet_b2_pmask50_futurepast
# export config_mod="sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"
# export config_mod="data.prob_mask_future=0.5 data.num_frames_future=5 sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2 data.prob_mask_cond=0.50"
export ckpt=530000

export version='DDPM'
export steps="100"

# export exp=../output/lvdm/smmnist_interp_big_c5t5f5_SPADE_lvdmv2
# export exp=../output/lvdm/smmnist_interp_big_c5t5f5_SPADE_lvdm
# export config_mod="data.num_frames_future=5 data.num_frames_cond=5 sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"


# export exp="../output/lvdm/smmnist_interp_big_c5t5f5_SPADE_lvdm-2"
export exp="../output/lvdm/smmnist_interp_big_c5t5f5_SPADE_lvdmv2-2"
export config_mod="model.spade=True data.num_frames_future=5 data.num_frames_cond=5 sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"


# export exp="../output/lvdm/smmnist_interp_big_c5t10f5_SPADE_lvdmv2"
# export exp="../output/lvdm/smmnist_interp_big_c5t10f5_SPADE_lvdm"
# export config_mod="model.spade=True data.num_frames_future=5 data.num_frames=10 model.spade=True sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore model.num_res_blocks=2"


cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py


CUDA_VISIBLE_DEVICES=5 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.preds_per_test=10 sampling.subsample=${steps} model.version=${version} --ckpt ${ckpt} --video_gen -v videos_${ckpt}_${version}_${steps}_nfp_${nfp}
