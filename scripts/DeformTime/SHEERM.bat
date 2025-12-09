@echo on
set MODEL=DeformTime
cd /d C:\Users\Administrator\Desktop\Code\DeformTime-main
call conda activate dtime

python run.py ^
  --is_training 1 ^
  --model_id SHEERM ^
  --model %MODEL% ^
  --data SHEERM ^
  --root_path ./data/dataset/ ^
  --data_path SHEERM_2_hid1.csv ^
  --features MS ^
  --target "Net Load" ^
  --freq 15min ^
  --seq_len 96 ^
  --label_len 48 ^
  --pred_len 96 ^
  --enc_in 31 ^
  --dec_in 31 ^
  --c_out 1 ^
  --patch_len 7 ^
  --stride 7 ^
  --batch_size 32 ^
  --train_epochs 10 ^
  --learning_rate 1e-4 ^
  --des sheerm ^
  --inverse
pause