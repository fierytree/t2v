export config="ucf101"
export data="../data/ucf101"
export exp="../output/lvdm/ucf10132_big288_4c4_pmask50_sqz8_sf_split2"
# export exp="../output/lvdm/ucf10132_big288_4c4_pmask50_lvdm2"
export nfp="16"
# export config_mod="data.prob_mask_cond=0.50 model.spade=True model.spade_dim=192 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32  sampling.batch_size=48 sampling.max_data_iter=1000 model.arch=unetmore"
export ckpt=930000

export config_mod="data.prob_mask_cond=0.50 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32  sampling.batch_size=60 sampling.max_data_iter=1000 model.arch=unetmore"

export version='DDPM'
export steps="100"


cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=7 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.preds_per_test=10 sampling.subsample=${steps} model.version=${version} --ckpt ${ckpt} --video_gen -v videos_${ckpt}_${version}_${steps}_nfp_${nfp}
