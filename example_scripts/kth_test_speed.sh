export config="kth64_big"
export data="../data/kth"

# export exp="../output/lvdm/kth64_big_5c10_sqzunet_performer3"
# export exp="../output/lvdm/kth64_big_5c10_sqzunet"
# export exp="../output/lvdm/kth64_big_5c10_unet"
# export exp="../output/lvdm/kth64_big_5c10_sqzunet_uneh"
export exp="../output/lvdm/kth64_big_5c10_sqzunet_performer2"
# export exp="../output/lvdm/kth64_big_5c10_sqzunetx4"
# export exp="../output/lvdm/kth64_big_5c10_sqzunetx8-2"
# export exp='../output/lvdm/kth64_big_5c10_sqzunetx4-shuffle'
# export exp='../output/lvdm/kth64_big_5c10_sqz4-shuffle2'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-shuffle-split'
# export exp='../output/lvdm/kth64_big_5c10_sqzunetx4-shuffle-seperable'
# export exp='../output/lvdm/kth64_big_5c10_sqz4-unscale'
# export exp='../output/lvdm/kth64_big_5c10_sqz5-sf-split-G3x3'
# export exp='../output/lvdm/kth64_big_5c10_sqz4-split'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split3'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-wt'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-wt2'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-wt3'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-wt4'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy2'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy3'
# export exp='../output/lvdm/kth64_big_5c10_sqz8-sf-split-dy6'


export config_mod="training.snapshot_freq=10000 sampling.num_frames_pred=30 data.num_frames=5 data.num_frames_cond=10 training.batch_size=64 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"

cp ${exp}/code/models/better/layerspp.py models/better/layerspp.py
cp ${exp}/code/models/better/ncsnpp_more.py models/better/ncsnpp_more.py


# mkdir -p ${exp}/code/models/better
# cp models/better/layerspp.py ${exp}/code/models/better/layerspp.py
# cp models/better/ncsnpp_more.py ${exp}/code/models/better/ncsnpp_more.py

CUDA_VISIBLE_DEVICES=7 python main.py --test_speed --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
