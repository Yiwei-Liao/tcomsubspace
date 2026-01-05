#!/bin/bash

# 数据集列表
datasets=("pubmed" "cora" "citeseer")

# MLP 隐藏层维度列表
hiddens=(2 4 8 16 32 64 128)

# 固定参数 (基于您提供的基准)
COMMON_ARGS="--AllSet_input_norm=True --All_num_layers=2 --Classifier_hidden=128 --dropout=0.7 --epochs=100 --heads=6 --init_hedge=avg --lr=0.001 --method=sheafHyperGNNDiag --runs=10 --sheaf_act=tanh --sheaf_normtype=sym_degree_norm --sheaf_pred_block=cp_decomp --wd=1e-05 --cuda=0"

echo "=== 开始运行维度变化测试 ==="

for dname in "${datasets[@]}"; do
    for hid in "${hiddens[@]}"; do
        echo "▶ Running: Dataset=${dname}, MLP_hidden=${hid}"
        
        python run_classification.py $COMMON_ARGS --dname $dname --MLP_hidden $hid
        
        echo "✔ Finished: Dataset=${dname}, MLP_hidden=${hid}"
        echo "----------------------------------------"
    done
done

echo "=== 所有维度测试完成 ==="