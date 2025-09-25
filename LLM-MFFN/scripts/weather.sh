export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=MFFN

for percent in 100
do
for pred_len in 96 192 336 720
do

python main.py \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.5
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 21 \
    --c_out 21 \
    --freq 0 \
    --lradj type4 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --is_gpt 1
    
done
done
