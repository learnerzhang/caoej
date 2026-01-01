#!/bin/bash
set -x
### 模型预测
echo "^^^predicting ***"
python -m models.kz_time.evaluate

python -m models.kz_dura.evaluate

python -m models.kz_num.evaluate

python -m models.kz_level.evaluate

python -m models.kz_comb_log.evaluate
