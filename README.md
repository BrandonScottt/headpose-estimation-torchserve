# headpose-estimation-torchserve

Model .pth link:
https://drive.google.com/file/d/1gguxIhg3TEnOYXDV6vWy1KZNuzWOmx2r/view?usp=drive_link

## Build TorchServe .mar Model Archive
torch-model-archiver \
  --model-name [model name] \
  --version 1.0 \
  --serialized-file 6DRepNet_300W_LP_BIWI.pth \
  --model-file model.py \
  --handler sixdrepnet_handler.py \
  --extra-files repvgg.py, se_block.py, utils.py, requirements.txt
