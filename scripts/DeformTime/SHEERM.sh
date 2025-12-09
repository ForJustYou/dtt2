export CUDA_VISIBLE_DEVICES=0
MODEL="DeformTime"
cd ../.. || exit  # 确保 cd 失败时退出
# 执行 Python 脚本
python -u run.py \
  --root_path ./data/dataset/ \
  --data_path SHEERM_2.csv \
  --model_id SHEERM \
  --model $MODEL \
  --data SHEERM \
  --features MS \
  --target "Net Load" \
  --freq 15min \
  --seq_len 336 \
  --label_len 0 \
  --d_model 32 \
  --pred_len 96 \
  --enc_in 28 \
  --dec_in 28 \
  --c_out 28 \
  --patch_len 24 \
  --n_heads 4 \
  --n_reshape 24 \
  --batch_size 32 \
  --train_epochs 100 \
  --dropout 0.25 \
  --e_layers 2 \
  --d_layers 2 \
  --kernel 7 \
  --patience 5 \
  --learning_rate 0.001 \
  --itr 1 \
  --des sheerm \
