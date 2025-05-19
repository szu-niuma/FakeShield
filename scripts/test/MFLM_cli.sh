WEIGHT_PATH=/home/yuyangxin/.cache/huggingface/hub/models--zhipeixu--fakeshield-v1-22b/snapshots/d0487aa9a8d7313c85e4cbef9de1e3a00fdc23c2
DTE_FDM_OUTPUT=./output
MFLM_OUTPUT=./MFLM_output
    
CUDA_VISIBLE_DEVICES=4 \
python /home/yuyangxin/data/FakeShield/MFLM_cli.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output ${DTE_FDM_OUTPUT} \
    --MFLM-output ${MFLM_OUTPUT} \
