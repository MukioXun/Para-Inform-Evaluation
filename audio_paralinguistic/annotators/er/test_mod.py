#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
model = "/home/u2023112559/qix/Models/Models/emotion2vec_plus_large"
model = AutoModel(
    model=model,
)
wav_file = f"/home/u2023112559/qix/Models/Models/emotion2vec_plus_large/example/test.wav"
res = model.generate(
    wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False
)
print(res)


#### output
# [{'key': 'test', 'labels': ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>'], 'scores': [1.0, 4.294446742658664e-12, 7.61923163450362e-12, 1.8145565605642844e-10, 7.18820003520193e-11, 1.369129571678453e-14, 9.774003828511013e-11, 8.875773449545932e-10, 5.661997978550056e-21]}]