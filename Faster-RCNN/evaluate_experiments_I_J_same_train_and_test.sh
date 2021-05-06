#!/usr/bin/bash

printf -v date '%(%Y-%m-%d)T' -1
outdir=same_train_and_test/evaluation_blur_0
model=output/blur_0/model.yaml
checkpoint=output/blur_0/model_final.pth
imagediri=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_I_0
imagedirj=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_J_0


# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I --imagedir $imagediri  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J --imagedir $imagedirj  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json


printf -v date '%(%Y-%m-%d)T' -1
outdir=same_train_and_test/evaluation_blur_1
model=output/blur_1/model.yaml
checkpoint=output/blur_1/model_final.pth
imagediri=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_I_1
imagedirj=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_J_1


# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I --imagedir $imagediri  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J --imagedir $imagedirj  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json


printf -v date '%(%Y-%m-%d)T' -1
outdir=same_train_and_test/evaluation_blur_2
model=output/blur_2/model.yaml
checkpoint=output/blur_2/model_final.pth
imagediri=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_I_2
imagedirj=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_J_2


# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I --imagedir $imagediri  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J --imagedir $imagedirj  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json


printf -v date '%(%Y-%m-%d)T' -1
outdir=same_train_and_test/evaluation_blur_4
model=output/blur_4/model.yaml
checkpoint=output/blur_4/model_final.pth
imagediri=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_I_4
imagedirj=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_J_4


# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I --imagedir $imagediri  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J --imagedir $imagedirj  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json




printf -v date '%(%Y-%m-%d)T' -1
outdir=same_train_and_test/evaluation_blur_8
model=output/blur_8/model.yaml
checkpoint=output/blur_8/model_final.pth
imagediri=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_I_8
imagedirj=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_J_8


# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I --imagedir $imagediri  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J --imagedir $imagedirj  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json


printf -v date '%(%Y-%m-%d)T' -1
outdir=same_train_and_test/evaluation_blur_16
model=output/blur_16/model.yaml
checkpoint=output/blur_16/model_final.pth
imagediri=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_I_16
imagedirj=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_J_16


# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I --imagedir $imagediri  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J --imagedir $imagedirj  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json



printf -v date '%(%Y-%m-%d)T' -1
outdir=same_train_and_test/evaluation_blur_32
model=output/blur_32/model.yaml
checkpoint=output/blur_32/model_final.pth
imagediri=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_I_32
imagedirj=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_J_32


# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I --imagedir $imagediri  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J --imagedir $imagedirj  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json



printf -v date '%(%Y-%m-%d)T' -1
outdir=same_train_and_test/evaluation_blur_64
model=output/blur_64/model.yaml
checkpoint=output/blur_64/model_final.pth
imagediri=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_I_64
imagedirj=/home/dimitar/neuro-project/Datasets/MSCOCO/testColor_blurimg_J_64


# Congruent vs Incongruent Exp I
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_I --imagedir $imagediri  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_I_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_I_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_I_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_I_coco_instances_results.json
rm ${outdir}/evaluation_*.json


# Congruent vs Incongruent Exp J
python test_recognition.py --dataset CONGRUENT_INCONGRUENT_EXP_J --imagedir $imagedirj  --outdir $outdir --model_yaml $model --checkpoint $checkpoint
mv ${outdir}/individual_scores.json ${outdir}/exp_J_individual_scores.json
mv ${outdir}/accuracies.json ${outdir}/exp_J_accuracies.json
mv ${outdir}/instances_predictions.pth ${outdir}/exp_J_instances_predictions.pth 
mv ${outdir}/coco_instances_results.json ${outdir}/exp_J_coco_instances_results.json
rm ${outdir}/evaluation_*.json



