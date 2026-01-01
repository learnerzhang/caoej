#!/bin/bash
set -x
### 模型训练
# 开闸时间
python -m models.kz_time.train
python -m models.kz_time.train_lr
python -m models.kz_time.train_svm
python -m models.kz_time.train_mlp
python -m models.kz_time.train_all_models

echo "^^^kz_time done ***\n"

# 目标水位
python -m models.kz_level.train
python -m models.kz_level.train_lr
python -m models.kz_level.train_svm
python -m models.kz_level.train_mlp
python -m models.kz_level.train_all_models

echo "^^^kz_level done ***\n"


# 开闸数量
python -m models.kz_num.train
python -m models.kz_num.train_lr
python -m models.kz_num.train_svm
python -m models.kz_num.train_mlp
python -m models.kz_num.train_all_models

echo "^^^kz_num done ***\n"

# 开闸时长
python -m models.kz_dura.train_cls
python -m models.kz_dura.train_cls_lr
python -m models.kz_dura.train_cls_svm
python -m models.kz_dura.train_cls_mlp
python -m models.kz_dura.train_all_models

echo "^^^kz_dura done ***\n"


# 开闸时长*开闸数量
python -m models.kz_comb_log.train
python -m models.kz_comb_log.train_lr
python -m models.kz_comb_log.train_svm
python -m models.kz_comb_log.train_mlp_g
python -m models.kz_comb_log.train_all_models

echo "^^^kz_comb_log done ***\n"
