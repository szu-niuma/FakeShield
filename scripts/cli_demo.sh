WEIGHT_PATH=/home/yuyangxin/.cache/huggingface/hub/models--zhipeixu--fakeshield-v1-22b/snapshots/d0487aa9a8d7313c85e4cbef9de1e3a00fdc23c2
IMAGE_PATH=/home/yuyangxin/data/dataset/columbia/Tp/canonxt_kodakdcs330_sub_20.tif
DTE_FDM_OUTPUT=./playground/DTE-FDM_output.jsonl
MFLM_OUTPUT=./playground/MFLM_output

# pip install -q transformers==4.37.2 >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=6 \
    python -m llava.serve.cli \
    --model-path ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --image-path ${IMAGE_PATH} \
    --output-path ${DTE_FDM_OUTPUT}

# pip install -q transformers==4.28.0 >/dev/null 2>&1
# CUDA_VISIBLE_DEVICES=0 \
#     python ./MFLM/cli_demo.py \
#     --version ${WEIGHT_PATH}/MFLM \
#     --DTE-FDM-output ${DTE_FDM_OUTPUT} \
#     --MFLM-output ${MFLM_OUTPUT}
