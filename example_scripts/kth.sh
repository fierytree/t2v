export config="kth64_big"
export data="../data/kth"

# export exp="../output/lvdm/kth64_big_5c10_sqzunet_performer2"
# export exp="../output/lvdm/kth64_big_5c10_sqzunet"
# export exp="../output/lvdm/kth64_big_5c10_unet"
# export exp="../output/lvdm/kth64_big_5c10_sqzunet_uneh"
# export exp="../output/lvdm/kth64_big_5c10_sqzunet_performer4"
# export exp='../output/lvdm/kth64_big_5c10_sqzunetx8'
# export exp='../output/lvdm/kth64_big_5c10_sqzunetx8-2'
# export exp='../output/lvdm/kth64_big_5c10_sqzunetx4'
# export exp='../output/lvdm/kth64_big_5c10_sqzunetx4-shuffle'
# export exp='../output/lvdm/kth64_big_5c10_sqz4-shuffle2'
# export exp='../output/lvdm/kth64_big_5c10_sqzunetx4-unshuffle'
# export exp='../output/lvdm/kth64_big_5c10_sqzunetx4-shuffle-seperable'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-shuffle-split'
# export exp='../output/lvdm/kth64_big_5c10_sqz4-unscale'
# export exp='../output/lvdm/kth64_big_5c10_sqz4-sf-split-G3x3'
# export exp='../output/lvdm/kth64_big_5c10_sqz5-sf-split-G3x3'
# export exp='../output/lvdm/kth64_big_5c10_sqz4-split'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split2'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-wt'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-wt2'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-wt3'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-wt4'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy2'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy3'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy4'
export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy5'
export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy6'

# export exp='../output/lvdm/kth64_big_5c10_sqz8-split-uneh'

export config_mod="training.snapshot_freq=10000 sampling.num_frames_pred=30 data.num_frames=5 data.num_frames_cond=10 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp="../output/lvdm/kth64_big_5c5f5_futmask50_general_unetm_lvdm"
# export exp="../output/lvdm/kth64_big_5c5f5_futmask50_general_sqz8_sf_split"
# export config_mod="training.snapshot_freq=10000 data.prob_mask_cond=0.50 data.prob_mask_future=0.50 sampling.num_frames_pred=20 data.num_frames=5 data.num_frames_cond=5 data.num_frames_future=5 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp="../output/lvdm/kth64_interp_big_c5t5f5_SPADE_lvdm"
# export exp="../output/lvdm/kth64_interp_big_c5t5f5_SPADE_lvdm-v2"
# export config_mod="model.spade=True model.spade_dim=128 training.snapshot_freq=10000 data.prob_mask_cond=0.0 data.prob_mask_future=0.50 sampling.num_frames_pred=20 data.num_frames=5 data.num_frames_cond=5 data.num_frames_future=5 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp='../output/lvdm/kth64_big_5c10_pmask50_lvdm'
# export config_mod="training.snapshot_freq=10000 sampling.num_frames_pred=30 data.num_frames=5 data.prob_mask_cond=0.50 data.num_frames_cond=10 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp="../output/lvdm/kth64_interp_big_c10t10f5_SPADE"
# export exp="../output/lvdm/kth64_interp_big_c10t10f5_SPADE_v2"
# export config_mod="model.spade=True model.spade_dim=128 training.snapshot_freq=10000 data.prob_mask_cond=0.0 data.prob_mask_future=0.50 sampling.num_frames_pred=20 data.num_frames=10 data.num_frames_cond=10 data.num_frames_future=5 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"


# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-warmup2'
# export config_mod="training.snapshot_freq=10000 sampling.num_frames_pred=30 data.num_frames=5 data.num_frames_cond=10 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore optim.warmup=50000 optim.lr=0.001"


cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py
CUDA_VISIBLE_DEVICES=7 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
