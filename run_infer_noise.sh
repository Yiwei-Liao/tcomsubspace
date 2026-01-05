#!/bin/bash

# 固定参数 (RHKH 链路预测基准)
BASE_CMD="python run_inference.py -score_func conve -opn corr -gpu 0 -epoch 400 -norm 2 -batch 256 -margin 0 \
-neg 400 -r2fc 0 -dist_type 0 -multi_r 1 -position 1 -shift 1 \
-gcn_drop 0.0 -id_drop 0.2 -data data/FB-AUTO"

# 噪声类型
noise_types=("gaussian" "rayleigh")

# 信噪比 (SNR) 列表
snrs=(-6 -4 -2 0 2 4 6 8 10)

echo "=== 开始运行推理任务噪声测试 ==="

for noise_type in "${noise_types[@]}"; do
  for snr in "${snrs[@]}"; do
    echo "▶ Running: Noise=${noise_type}, SNR=${snr}dB"
    
    # 每次运行都会重新训练并评估指定噪声下的性能
    $BASE_CMD --add_noise --noise_type $noise_type --snr $snr

    echo "✔ Finished: Noise=${noise_type}, SNR=${snr}dB"
    echo "----------------------------------------"
  done
done

echo "=== 推理任务噪声测试完成 ==="