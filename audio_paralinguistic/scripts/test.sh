cd /home/u2023112559/qix/Project/Audio_Captior/audio_paralinguistic/scripts
# python run_test_batch.py \
#     --input /home/u2023112559/qix/datasets/PASM_Lite \
#     --output /home/u2023112559/qix/Project/Audio_Captior/audio_paralinguistic/data/experiments/exp_230320_SUM_ANALYSIS \
#     --limit 200 \
#     --skip-tone \
#     --skip-age \

python run_sar_audio_reasoner.py \
    --input /home/u2023112559/qix/Project/Audio_Captior/audio_paralinguistic/data/experiments/exp_230320_SUM_ANALYSIS/merged \
    --audio-dirs /home/u2023112559/qix/datasets/PASM_Lite \
    --limit 200 \

# ./run_pipeline_with_tone.sh \
#     --input /home/u2023112559/qix/datasets/PASM_Lite \
#     --output /home/u2023112559/qix/Project/Audio_Captior/audio_paralinguistic/data/experiments/test_new_pipeline_output \
#     --limit 5