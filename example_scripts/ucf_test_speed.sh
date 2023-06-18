export config="ucf101"
export data="../data/ucf101"

# export exp="../output/lvdm/ucf10132_big288_4c4_pmask50_sqz8_sf_split2"
# export exp="../output/ucf10132_big192_288_4c4_pmask50_unetm_spade"
# export config_mod="model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# export exp="../output/lvdm/ucf10132_big288_4c4_pmask50_sqz8_sf_split2"
# export exp="../output/lvdm/ucf10132_big288_4c4_pmask50_lvdm2"
export exp="../output/lvdm/ucf10132_big288_4c4_pmask50_unet"
export config_mod="model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"


cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=5 python main.py --test_speed --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
