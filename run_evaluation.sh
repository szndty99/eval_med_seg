#!/bin/bash

# 医学图像分割评估脚本
# 用于运行evaluate_segmentation.py，评估分割结果

# 默认参数
PRED_DIR=""
GT_DIR=""
OUTPUT_DIR=""
PROB_DIR=""
THRESHOLD=0.5
SAVE_INDIVIDUAL=false

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -p, --pred_dir DIR       分割结果目录（必需）"
    echo "  -g, --gt_dir DIR         真值标签目录（必需）"
    echo "  -o, --output_dir DIR     输出目录（必需）"
    echo "  -r, --prob_dir DIR       概率图目录（可选，用于ROC和PR曲线）"
    echo "  -t, --threshold VALUE    分割阈值（默认: 0.5）"
    echo "  -i, --save_individual    保存每个案例的单独评估结果"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "说明:"
    echo "  该脚本现在支持处理预测文件与真值文件命名不一致的情况，"
    echo "  只要文件名中包含相同的数字标识（如0181）即可自动匹配。"
    echo ""
    echo "示例:"
    echo "  $0 -p /path/to/predictions -g /path/to/ground_truth -o /path/to/output"
    echo "  $0 -p /path/to/predictions -g /path/to/ground_truth -o /path/to/output -r /path/to/probability_maps -t 0.6 -i"
    echo ""
    echo "预设配置:"
    echo "  $0 --preset binary       二分类分割评估预设"
    echo "  $0 --preset multi        多类别分割评估预设"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--pred_dir)
            PRED_DIR="$2"
            shift
            shift
            ;;
        -g|--gt_dir)
            GT_DIR="$2"
            shift
            shift
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -r|--prob_dir)
            PROB_DIR="$2"
            shift
            shift
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift
            shift
            ;;
        -i|--save_individual)
            SAVE_INDIVIDUAL=true
            shift
            ;;
        --preset)
            PRESET="$2"
            shift
            shift
            
            # 预设配置
            if [[ "$PRESET" == "binary" ]]; then
                # 二分类分割评估预设
                if [[ -z "$PRED_DIR" ]]; then
                    PRED_DIR="/groupshare/data/nnUNet_raw_data_base/nnUNet_inference/Task001_BinaryClass/results"
                fi
                if [[ -z "$GT_DIR" ]]; then
                    GT_DIR="/groupshare/data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_BinaryClass/labelsTs"
                fi
                if [[ -z "$OUTPUT_DIR" ]]; then
                    OUTPUT_DIR="/groupshare/data/3D-TransUNet-main/evaluation_results/binary_$(date +%Y%m%d_%H%M%S)"
                fi
                if [[ -z "$PROB_DIR" ]]; then
                    PROB_DIR="/groupshare/data/nnUNet_raw_data_base/nnUNet_inference_npz/Task001_BinaryClass/results"
                fi
                THRESHOLD=0.5
                SAVE_INDIVIDUAL=true
            elif [[ "$PRESET" == "multi" ]]; then
                # 多类别分割评估预设
                if [[ -z "$PRED_DIR" ]]; then
                    PRED_DIR="/groupshare/data/nnUNet_raw_data_base/nnUNet_inference/Task002_MultiClass/results"
                fi
                if [[ -z "$GT_DIR" ]]; then
                    GT_DIR="/groupshare/data/nnUNet_raw_data_base/nnUNet_raw_data/Task002_MultiClass/labelsTs"
                fi
                if [[ -z "$OUTPUT_DIR" ]]; then
                    OUTPUT_DIR="/groupshare/data/3D-TransUNet-main/evaluation_results/multi_$(date +%Y%m%d_%H%M%S)"
                fi
                if [[ -z "$PROB_DIR" ]]; then
                    PROB_DIR="/groupshare/data/nnUNet_raw_data_base/nnUNet_inference_npz/Task002_MultiClass/results"
                fi
                THRESHOLD=0.5
                SAVE_INDIVIDUAL=true
            else
                echo "错误: 未知的预设配置 '$PRESET'"
                show_help
                exit 1
            fi
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知选项 '$key'"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$PRED_DIR" || -z "$GT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "错误: 缺少必需参数"
    show_help
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建命令
CMD="python ./evaluate_segmentation.py \
    --pred_dir \"$PRED_DIR\" \
    --gt_dir \"$GT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --threshold $THRESHOLD"

# 添加可选参数
if [[ ! -z "$PROB_DIR" ]]; then
    CMD="$CMD --prob_dir \"$PROB_DIR\""
fi

if [[ "$SAVE_INDIVIDUAL" = true ]]; then
    CMD="$CMD --save_individual"
fi

# 显示将要执行的命令
echo "执行命令:"
echo "$CMD"
echo ""

# 执行命令
eval $CMD

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo "评估完成！结果已保存到: $OUTPUT_DIR"
    echo "生成的图表:"
    echo "- $OUTPUT_DIR/iou_accuracy_curve.png"
    echo "- $OUTPUT_DIR/dice_distribution.png"
    echo "- $OUTPUT_DIR/metrics_boxplot.png"
    echo "- $OUTPUT_DIR/volume_comparison.png"
    
    if [[ ! -z "$PROB_DIR" ]]; then
        echo "- $OUTPUT_DIR/roc_curve.png"
        echo "- $OUTPUT_DIR/pr_curve.png"
    fi
    
    echo ""
    echo "平均指标:"
    cat "$OUTPUT_DIR/average_metrics.txt"
else
    echo "评估失败！请检查错误信息。"
fi 