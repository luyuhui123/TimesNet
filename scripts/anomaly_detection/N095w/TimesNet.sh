export CUDA_VISIBLE_DEVICES=1

# python -u run.py \
#   --task_name anomaly_detection \
#   --is_training 1 \
#   --root_path ./dataset/SMD/N095w \
#   --model_id N095w \
#   --model TimesNet \
#   --data N095w \
#   --features M \
#   --seq_len 100 \
#   --pred_len 0 \
#   --d_model 64 \
#   --d_ff 64 \
#   --e_layers 2 \
#   --enc_in 3 \
#   --c_out 3 \
#   --top_k 4 \
#   --anomaly_ratio 0.77 \
#   --batch_size 128 \
#   --train_epochs 20


  python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/SMD/N095w \
  --model_id N095w \
  --model TimesNet \
  --n_features 3 \
  --alpha 0.2 \
  --feat_gat_embed_dim -1\
  --use_gatv2 True\
  --data N095w \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --top_k 4 \
  --anomaly_ratio 0.77 \
  --batch_size 128 \
  --train_epochs 20