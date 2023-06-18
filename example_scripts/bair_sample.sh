export config="bair_big"
export data="../data/bair"
export nfp="15"

export exp="../output/ijcai23/bair64_big192_5c1_pmask50_spynet"
# export exp=../output/lvdm/bair64_big192_5c1_pmask50_sqz8_sf_split
# export exp=../output/lvdm/bair64_big192_5c1_pmask50_lvdm
export config_mod="data.prob_mask_cond=0.50 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=1 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"

# export exp="../output/ijcai23/bair64_big192_5c1_prevfutpmask50_spynet"
# export config_mod="model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 data.num_frames=5 data.num_frames_cond=1 data.num_frames_future=1 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"


# export exp="../output/ijcai23/bair64_big192_5c1_pmask50_spynet"
# export config_mod="model.ngf=192 model.n_head_channels=192 data.prob_mask_cond=0.50 data.prob_mask_future=0.5 data.num_frames=5 data.num_frames_cond=2 data.num_frames_future=2 training.batch_size=64 sampling.batch_size=100 sampling.max_data_iter=1000 model.arch=unetmore"

# export exp="../output/lvdm/bair64_big192_5c2_pmask50_lvdmv2"
# export exp="../output/lvdm/bair64_big192_5c2_pmask50_lvdm"
# export config_mod="data.prob_mask_cond=0.50 model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=64  sampling.batch_size=200 sampling.max_data_iter=1000 model.arch=unetmore"


export ckpt=320000
export version='DDPM'
export steps="100"

cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

# for ckpt in {11..12}
# do
#     ((ckpt *= 10000))
#     echo $ckpt
#     CUDA_VISIBLE_DEVICES=5 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.max_data_iter=1 sampling.preds_per_test=1 sampling.subsample=${steps} model.version=${version} --ckpt ${ckpt} --video_gen -v videos_${ckpt}_${version}_${steps}_nfp_${nfp}_1batch
# done

CUDA_VISIBLE_DEVICES=9 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.preds_per_test=10 sampling.subsample=${steps} model.version=${version} --ckpt ${ckpt} --video_gen -v videos_${ckpt}_${version}_${steps}_nfp_${nfp}