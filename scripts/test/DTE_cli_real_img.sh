WEIGHT_PATH=/home/yuyangxin/.cache/huggingface/hub/models--zhipeixu--fakeshield-v1-22b/snapshots/d0487aa9a8d7313c85e4cbef9de1e3a00fdc23c2
IMAGE_PATH=./playground/image/Sp_D_CRN_A_ani0043_ani0041_0373.jpg
DTE_FDM_OUTPUT=./playground/MagicBrush_
MFLM_OUTPUT=./playground/MFLM_output

CUDA_VISIBLE_DEVICES=6 \
python /home/yuyangxin/data/FakeShield/DTE_cli.py \
    --model-path ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --image-path ${IMAGE_PATH} \
    --target_type "real_img" \
    --output-path ${DTE_FDM_OUTPUT}
    
