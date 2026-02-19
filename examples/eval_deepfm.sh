nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --cudabacktrace=all \
    -o deepfm_full_profile \
    python run_deepfm.py