export config="cityscapes_big"
export data="../data/cityscapes"
export devices="0"
export nfp="28"

export exp="../output/lvdm/city32_big192_5c2_lvdm"
export ckpt="1220000"
export config_mod="model.ngf=192 model.n_head_channels=192 data.num_frames=5 data.num_frames_cond=2 training.batch_size=32 sampling.batch_size=35 sampling.max_data_iter=1000 model.arch=unetmore"


export version='DDPM'
export steps="100"



cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=7 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod} sampling.num_frames_pred=${nfp} sampling.preds_per_test=10 sampling.subsample=${steps} model.version=${version} --ckpt ${ckpt} --video_gen -v videos_${ckpt}_${version}_${steps}_nfp_${nfp}
