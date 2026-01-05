#!/bin/bash

# 数据集列表
datasets=("pubmed" "cora" "citeseer")

# 噪声类型
noise_types=("gaussian" "rayleigh")

# 固定参数 (这里设定一个标准的 MLP_hidden，例如 32)
COMMON_ARGS="--AllSet_input_norm=True --All_num_layers=2 --MLP_hidden=32 --Classifier_hidden=128 --dropout=0.7 --epochs=100 --heads=6 --init_hedge=avg --lr=0.001 --method=sheafHyperGNNDiag --runs=10 --sheaf_act=tanh --sheaf_normtype=sym_degree_norm --sheaf_pred_block=cp_decomp --wd=1e-05 --cuda=0"

echo "=== 开始运行分类任务噪声测试 ==="

for dname in "${datasets[@]}"; do
    for noise in "${noise_types[@]}"; do
        echo "▶ Running: Dataset=${dname}, Noise=${noise} (SNR Sweep -6 to 10dB)"
        
        # 注意：这里添加了 --add_noise 开关
        python run_classification.py $COMMON_ARGS --dname $dname --add_noise --noise_type $noise
        
        echo "✔ Finished: Dataset=${dname}, Noise=${noise}"
        echo "----------------------------------------"
    done
done

echo "=== 分类任务噪声测试完成 ==="