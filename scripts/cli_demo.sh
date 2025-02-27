WEIGHT_PATH=zhipeixu/fakeshield-v1-22b
IMAGE_PATH=./playground/image/Sp_D_CRN_A_ani0043_ani0041_0373.jpg
DTE_FDM_OUTPUT=./playground/DTE-FDM_output.jsonl
MFLM_OUTPUT=./playground/MFLM_output
CUDA_VISIBLE_DEVICES=6 \
python -m ./DTE-FDM/llava/serve/cli.py \
    --model-path  ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --image-path ${IMAGE_PATH} \
    --output-path ${DTE_FDM_OUTPUT}

pip install -q transformers==4.28.0  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=6 \
python ./MFLM/cli_demo.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output ${DTE_FDM_OUTPUT} \
    --MFLM-output ${MFLM_OUTPUT}
