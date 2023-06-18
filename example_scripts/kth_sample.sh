export config="kth64_big"
export data="../data/kth"
export devices="0"
export ckpt="490000"
export nfp="30"

export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy2'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-shuffle-split'
# export exp="../output/lvdm/kth64_big_5c10_sqz4-split"
# export exp="../output/lvdm/kth64_big_5c5f5_futmask50_general_unetm_lvdm"
# export exp="../output/lvdm/kth64_big_5c5f5_futmask50_general_sqz8_sf_split"
# export exp="../output/lvdm/kth64_interp_big_c5t5f5_SPADE_lvdm"
# export exp="../output/lvdm/kth64_interp_big_c5t5f5_SPADE_lvdm-v2"

export config_mod="data.num_frames=5 data.num_frames_future=0 data.num_frames_cond=10 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
# export config_mod="data.prob_mask_cond=0.50 data.prob_mask_future=0.50 data.num_frames=5 data.num_frames_future=5 data.num_frames_cond=5 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"
# export config_mod="model.spade=True model.spade_dim=128 data.prob_mask_cond=0.0 data.prob_mask_future=0.50 data.num_frames=5 data.num_frames_future=5 data.num_frames_cond=5 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"

# export exp="../output/lvdm/kth64_interp_big_c10t10f5_SPADE"
# export exp="../output/lvdm/kth64_interp_big_c10t10f5_SPADE_v2"
# export config_mod="model.spade=True model.spade_dim=128 data.prob_mask_cond=0.0 data.prob_mask_future=0.50 data.num_frames=10 data.num_frames_future=5 data.num_frames_cond=10 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"


export version='DDPM'
export steps="100"

cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=6 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.preds_per_test=10 sampling.subsample=${steps} model.version=${version} --ckpt ${ckpt} --video_gen -v videos_${ckpt}_${version}_${steps}_nfp_${nfp}
