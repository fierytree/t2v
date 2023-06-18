export config="ucf101"
export data="../data/ucf101"

export exp='../output/ijcai23/ucf10132_big288_4c4_pmask50_spynet_cross2'
# export exp='../output/ijcai23/ucf10132_big288_4c4_pmask50_spynet_cross'
export config_mod="training.snapshot_freq=10000 model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=1 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

# exp="../output/lvdm/ucf10132_big192_288_4c4_pmask50_sqzunet_performer_spade3"
# config_mod="training.snapshot_freq=10000 model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}