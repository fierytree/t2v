export config="bair_big"
export data="../data/bair"
export devices="0,1,2,3"

# export exp="../output/lvdm/bair64_big192_5c1_prevfutpmask50_sqz8_sf_split"
# export exp="../output/lvdm/bair64_big192_5c1_prevfutpmask50_sqz8_sf_split2"
export exp="../output/lvdm/bair64_big192_5c1_prevfutpmask50_sqz8_sf_split3"
# export exp="../output/lvdm/bair64_big192_5c1_prevfutpmask50_sqzunet_performer"
export config_mod="training.snapshot_freq=10000 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 data.num_frames_future=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp2="../output/bair64_big192_5c1_prevfutpmask50_unetm_general"
# export config_mod="training.snapshot_freq=10000 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 data.num_frames_future=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp="../output/bair64_big192_5c1_unetm"
# export exp="../output/lvdm/bair64_big192_5c1_sqz8_sf_split"
# export config_mod="training.snapshot_freq=10000 model.ngf=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp="../output/bair64_big192_5c1_unetm_spade"
# export config_mod="training.snapshot_freq=10000 model.spade=True model.spade_dim=128 model.ngf=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp="../output/lvdm/bair64_big192_5c1_pmask50_lvdm"
# export config_mod="training.snapshot_freq=10000 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"


# cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
# cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py


mkdir -p ${exp}/code/models/better
cp models/better/layerspp.py ${exp}/code/models/better/layerspp.py
cp models/better/ncsnpp_more.py ${exp}/code/models/better/ncsnpp_more.py

# export exp="../output/bair64_big192_5c2_unetm5"
# export config_mod="training.snapshot_freq=10000 model.ngf=192 model.n_head_channels=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
CUDA_VISIBLE_DEVICES=5 python main.py --test_speed --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
