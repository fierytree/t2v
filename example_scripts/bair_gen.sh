export config="bair_big"
export data="../data/bair"
export devices="0,1,2,3"

# export exp="../output/bair64_big192_5c2_unetm5"
# export config_mod="training.snapshot_freq=10000 model.ngf=192 model.n_head_channels=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

export exp="../output/lvdm/bair64_big192_5c1_pmask50_lvdm"
export config_mod="training.snapshot_freq=10000 model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=6,7  python main.py --resume_training --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
