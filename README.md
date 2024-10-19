# Experiments on OSCAR

## Download

**Note**

It is recommended to download large files with [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) for faster speed. For convenience, the AzCopy executable tool has been downloaded and unzip to the root directory of the project in advance.

**Datasets**

Run command below to obtain the extracted image region features, object tags, and the original text annotations for each downstream tasks, which are provided by [Oscar](https://github.com/microsoft/Oscar).

```
wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1

# VQA
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/vqa.zip oscar/datasets/
unzip oscar/datasets/vqa.zip -d oscar/datasets/
# NLVR2
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/nlvr2.zip oscar/datasets/
unzip oscar/datasets/nlvr2.zip -d oscar/datasets/
# Image-Text Retrieval
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/coco_ir.zip oscar/datasets/
unzip oscar/datasets/coco_ir.zip -d oscar/datasets/
```

**Pre-trained Models**

Run command below to obtain the pre-trained models, which are provided by [Oscar](https://github.com/microsoft/Oscar).

```
# OSCAR-base
./azcopy copy https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/base-vg-labels.zip oscar/pretrained_models
unzip oscar/pretrained_models/base-vg-labels.zip -d oscar/pretrained_models/
```

## Fine-tuning

Run command below to obtain the fine-tuned teacher for each task.

```
# VQA
python oscar/run_vqa.py \
--task_name vqa \
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--model_name_or_path oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/vqa/teacher \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--learning_rate 5e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear

# NLVR2
python oscar/run_nlvr.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--model_name_or_path oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/nlvr/teacher \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 20 \
--learning_rate 3e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2

# Image-Text Retrieval
python oscar/run_retrieval.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--model_name_or_path oscar/pretrained_models/base-vg-labels/ep_67_588997 \
--output_dir oscar/model/coco_ir/teacher \
--do_train \
--do_lower_case \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--learning_rate 2e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_steps 5000 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.1 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt
```

## Distillation

Run command below to obtain distilled student for each task.

```
# VQA
# FT
python oscar/run_vqa_with_ft.py \
--task_name vqa \
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--model_name_or_path oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/vqa/student \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--learning_rate 5e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear \
--num_hidden_layers 6
# KD
python oscar/run_vqa_with_kd.py \
--task_name vqa \
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--teacher_model oscar/model/vqa/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/vqa/student \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--learning_rate 5e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 --drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 
# PKD
python oscar/run_vqa_with_pkd.py 
--task_name vqa 
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--teacher_model oscar/model/vqa/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/vqa/student \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--learning_rate 5e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 
--loss_type bce 
--classifier linear 
--num_hidden_layers 6 
--alpha 0.5 
--temperature 5.0 
--beta 500
# TD
python oscar/run_vqa_with_td.py \
--task_name vqa \
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--teacher_model oscar/model/vqa/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/vqa/student \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--learning_rate 5e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# EMD
python oscar/run_vqa_with_emd.py \
--task_name vqa \
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--teacher_model oscar/model/vqa/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/vqa/student \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--learning_rate 5e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# MGSKD
python oscar/run_vqa_with_mgskd.py \
--task_name vqa \
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--teacher_model oscar/model/vqa/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/vqa/student \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--learning_rate 5e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# MMKD
python oscar/run_vqa_with_mmkd.py \
--task_name vqa \
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--teacher_model oscar/model/vqa/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/vqa/student \
--do_train \
--do_lower_case \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 25 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type bce \
--classifier linear \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--teacher_learning_rate 5e-5 \
--student_learning_rate 5e-5 \
--strategy skip \
--beta1 0.5 \
--beta2 0.25 \
--beta3 0.25

# NLVR2
# FT
python oscar/run_nlvr_with_ft.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--model_name_or_path oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/nlvr/student \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 20 \
--learning_rate 3e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--num_hidden_layers 6
# KD
python oscar/run_nlvr_with_kd.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--teacher_model oscar/model/nlvr/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/nlvr/student \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 20 \
--learning_rate 3e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0
# PKD
python oscar/run_nlvr_with_pkd.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--teacher_model oscar/model/nlvr/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/nlvr/student \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 20 \
--learning_rate 3e-5 \
--evaluate_during_training \
--logging_steps 50\ 
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 500
# TD
python oscar/run_nlvr_with_td.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--teacher_model oscar/model/nlvr/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/nlvr/student \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 20 \
--learning_rate 3e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# EMD
python oscar/run_nlvr_with_emd.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--teacher_model oscar/model/nlvr/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/nlvr/student \
--do_train \
--do_lower_case --max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 20 \
--learning_rate 3e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# MGSKD
python oscar/run_nlvr_with_mgskd.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--teacher_model oscar/model/nlvr/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \
--output_dir oscar/model/nlvr/student \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 20 \
--learning_rate 3e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# MMKD
python oscar/run_nlvr_with_mmkd.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--teacher_model oscar/model/nlvr/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_107_1192087 \ 
--output_dir oscar/model/nlvr/student \
--do_train \
--do_lower_case \
--max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 20 \
--evaluate_during_training \
--logging_steps 50 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.3 \
--weight_decay 0.05 \
--warmup_steps 10000 \
--loss_type ce \
--classifier mlp \
--cls_hidden_scale 3 \
--num_choice 2 \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--teacher_learning_rate 3e-5 \
--student_learning_rate 3e-5 \
--strategy skip \
--beta1 0.5 \
--beta2 0.25 \
--beta3 0.25

# Image-Text Retrieval
# FT
python oscar/run_retrieval_with_ft.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--model_name_or_path oscar/pretrained_models/base-vg-labels/ep_67_588997 \
--output_dir oscar/model/coco_ir/teacher \
--do_train \
--do_lower_case \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--learning_rate 2e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_steps 5000 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.1 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--num_hidden_layers 6
# KD
python oscar/run_retrieval_with_kd.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--teacher_model oscar/model/coco_ir/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_67_588997 \
--output_dir oscar/model/coco_ir/teacher \
--do_train \
--do_lower_case \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--learning_rate 2e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_steps 5000 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.1 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0
# PKD
python oscar/run_retrieval_with_pkd.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--teacher_model oscar/model/coco_ir/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_67_588997 \
--output_dir oscar/model/coco_ir/teacher \
--do_train \
--do_lower_case \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--learning_rate 2e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_steps 5000 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.1 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 500
# TD
python oscar/run_retrieval_with_td.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--teacher_model oscar/model/coco_ir/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_67_588997 \
--output_dir oscar/model/coco_ir/teacher \
--do_train \
--do_lower_case \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--learning_rate 2e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_steps 5000 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.1 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# EMD
python oscar/run_retrieval_with_emd.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--teacher_model oscar/model/coco_ir/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_67_588997 \
--output_dir oscar/model/coco_ir/teacher \
--do_train \
--do_lower_case \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--learning_rate 2e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_steps 5000 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.1 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# MGSKD
python oscar/run_retrieval_with_mgskd.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--teacher_model oscar/model/coco_ir/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_67_588997 \
--output_dir oscar/model/coco_ir/teacher \
--do_train \
--do_lower_case \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--learning_rate 2e-5 \
--evaluate_during_training \
--logging_steps 50 \
--save_steps 5000 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.1 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--beta 0.01
# MMKD
python oscar/run_retrieval_with_mmkd.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--teacher_model oscar/model/coco_ir/teacher \
--student_model oscar/pretrained_models/base-vg-labels/ep_67_588997 \
--output_dir oscar/model/coco_ir/teacher \
--do_train \
--do_lower_case \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--num_train_epochs 30 \
--evaluate_during_training \
--logging_steps 50 \
--save_steps 5000 \
--save_epoch 1 \
--seed 88 \
--drop_out 0.1 \
--weight_decay 0.05 \
--warmup_steps 0 \
--loss_type ce \
--classifier linear \
--num_captions_per_img_val 20 \
--eval_caption_index_file minival_caption_indexs_top20.pt \
--num_hidden_layers 6 \
--alpha 0.5 \
--temperature 5.0 \
--teacher_learning_rate 2e-5 \
--student_learning_rate 2e-5 \
--strategy skip \
--beta1 0.5 \
--beta2 0.25 \
--beta3 0.25
```

## Inference

Run command below to obtain predictions of the distilled student model for each task.

```
# VQA 
python oscar/run_vqa.py \
--task_name vqa \
--data_dir oscar/datasets/vqa/2k \
--model_type oscar \
--model_name_or_path oscar/model/vqa/student \
--output_dir oscar/model/vqa/student \
--do_test \
--max_seq_length 128 \
--max_seq_length 50 \
--per_gpu_eval_batch_size 32

# NLVR2
python oscar/run_nlvr.py \
--task_name nlvr \
--data_dir oscar/datasets/nlvr2/ft_corpus \
--model_type oscar \
--model_name_or_path oscar/model/nlvr/student \
--output_dir oscar/model/nlvr/student \
--do_test \
--max_seq_length 55 \
--max_img_seq_length 40 \
--per_gpu_eval_batch_size 32

# Image-Text Retrieval
python oscar/run_retrieval.py \
--task_name coco_ir \
--data_dir oscar/datasets/coco_ir \
--model_type oscar \
--model_name_or_path oscar/model/coco_ir/student \
--output_dir oscar/model/coco_ir/student \
--do_test \
--max_seq_length 70 \
--max_img_seq_length 50 \
--per_gpu_eval_batch_size 32 \
--num_captions_per_img_val 5 \
--cross_image_eval \
--eval_img_keys_file test_img_keys.tsv
```
