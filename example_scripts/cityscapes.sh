export config="cityscapes_big"
export data="../data/cityscapes"

# Video prediction
# export exp="../output/lvdm/city32_big192_5c2_lvdm"
# export exp="../output/lvdm/city32_big192_5c2_sqz8_sf_split"
export exp="../output/lvdm/city32_big192_5c2_v3"
export config_mod="training.snapshot_freq=10000 model.ngf=192 model.n_head_channels=192 sampling.num_frames_pred=28 data.num_frames=5 data.num_frames_cond=2  training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=2,3 python main.py --resume_training --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
