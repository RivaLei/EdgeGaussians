#!/bin/bash
# set -x  # 启用调试模式
# # 捕获错误并打印错误信息
# trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# COMPUT_FORCE=FALSE

# 指定ABC数据 测试 改进后的 occlusion loss 【gt 原始方法 真值】

# Define the paths to the scripts
#不能有空格
ROOT_DIR="/media/wuhan-ds/D2/lz/EdgeGaussians"
TRAIN_SCRIPT="${ROOT_DIR}/train_gaussians.py"
FIT_SCRIPT="${ROOT_DIR}/fit_edges.py"
VISUALIZE_SCRIPT="${ROOT_DIR}/visualize_points_with_major_dirs.py"
EVAL_SCRIPT="${ROOT_DIR}/eval_general.py"

#-----
#不同数据集需要更改的路径
#从命令行中读取--暂时失败

GT_BASE_DIR="${ROOT_DIR}/data/ABC-NEF_Edge/groundtruth"
DATASETNAME="ABC"
SCENE_NAME="00004926"
TEST_VIEW_ID="0" #设置典型的view id, 用于对比# 这个放在config里面--todo


# # 解析命令行参数
# while getopts g:d:s:v: flag
# do
#     case "${getopts}" in
#         g) GT_BASE_DIR=${ROOT_DIR}${OPTARG};;
#         d) DATASETNAME=${OPTARG};;
#         s) SCENE_NAME=${OPTARG};;
#         v) TEST_VIEW_ID=${OPTARG};;
#     esac
# done

# 打印变量值以验证
echo "GT_BASE_DIR: $GT_BASE_DIR"
echo "DATASETNAME: $DATASETNAME"
echo "SCENE_NAME: $SCENE_NAME"
echo "TEST_VIEW_ID: $TEST_VIEW_ID"


#------------------------------------------------
# 原始方法
CONFIG_PATH_3DGS="${ROOT_DIR}/configs/ABC_DexiNed.json"
# 读取 CONFIG_PATH_3DGS JSON 文件 根据关键字获取信息
EDGE_DETECTION_METHOD=$(jq -r '.data.edge_detection_method' "$CONFIG_PATH_3DGS")
VERSION_NAME="release"
INPUT_PLY_3DGS="${ROOT_DIR}/output/${DATASETNAME}/${VERSION_NAME}_${EDGE_DETECTION_METHOD}/${SCENE_NAME}/pts_with_major_dirs.ply"



echo "Edge Detection Method: $EDGE_DETECTION_METHOD"

# Execute the scripts
echo "[EDGS]:1-Running train_gaussians.py..."
#todo 输出最终loss[total projection dir ratio]
#todo 添加verbose输出info级别--脚本执行时 只输出关键信息
#todo 针对config中指定视角 输出render视图
python3 $TRAIN_SCRIPT "--config_file" $CONFIG_PATH_3DGS "--scene_name" $SCENE_NAME      

echo "[EDGS]:2-Running fit_edgs.py..."
python3 $FIT_SCRIPT "--config_file" $CONFIG_PATH_3DGS "--scene_name" $SCENE_NAME "--save_filtered" "--save_sampled_points"

echo "[EDGS]:3-Running visualize_points_with_major_dirs.py..."
python3 $VISUALIZE_SCRIPT "--input_ply" $INPUT_PLY_3DGS 

# 与gt对比

echo "[EDGS]:4-Running eval_general.py..."
                
python3 $EVAL_SCRIPT \
        "--dataset" $DATASETNAME \
        "--scan_names" $SCENE_NAME \
        "--version" $VERSION_NAME \
        "--edge_detector" $EDGE_DETECTION_METHOD \
        "--gt_base_dir" $GT_BASE_DIR \
        "--use_parametric_edges" \
        "--visualize" \
        "--write_metrics"

        
echo "[EDGS]:All scripts executed successfully."
#------------------------------------------------


#------------------------------------------------
# 将原始方法和改进后的方法的结果进行对比
# 两个方法的loss变化--根据tensorboard的输出
TENSORBOARD_PATH="${ROOT_DIR}/logs/${DATASETNAME}/release_DexiNed/${SCENE_NAME}/"


cd $TENSORBOARD_PATH
tensorboard --logdir=$TENSORBOARD_PATH
#设置一个停顿 用于查看tensorboard的输出 按键 ctrl+c 继续执行
read -p "Press any key to continue... " -n1 -s


#将原始方法和改进后的方法的render结果进行对比[指定视角]--todo 还没执行