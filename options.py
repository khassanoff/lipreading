gpu = '5'
random_seed = 0
data_type = 'unseen'
video_path = 'dataset/GRID/lip/'
train_list = f'data/{data_type}_train.txt'
val_list = f'data/{data_type}_val.txt'
anno_path = 'dataset/GRID/GRID_align_txt'
vid_padding = 75
txt_padding = 200
batch_size = 200
drop = 0.5
base_lr = 1e-4
weight_decay = 1e-8
patience = 15 
num_workers = 16
max_epoch = 1000
display = 25
test_step = 500
save_prefix = f'weights/LipNet_{data_type}'
is_optimize = True

#weights = 'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'
