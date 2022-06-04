# Radiology Image Classification Competition Code
We achieved first place on the machine learning competition on classifying radiology images.

Team Unemployed Pony Spuds
- Kai Nakamura
- Jerry Huang 
- Madeline Shao
- Sneh Patel 
- Christian Zapata-Sanin

## Overview

The goal of the project is to classify radiology images in 14 disease
categories. 

We achieve the best average MSE score across the disease categories (0.666 on solution set).

Competition overview can be found [here](https://cs156.caltech.edu/web/challenges/challenge-page/24/overview)

This was part of CS156 Spring 2022 course.

*Disclaimer from instructor: Future CS156b students should not look at the code while taking the course*

## Results

We achieve [an average MSE of 0.666 on solution set](https://cs156.caltech.edu/web/challenges/challenge-page/24/leaderboard)

Detailed results can be found [here](https://docs.google.com/spreadsheets/d/19EHmtVX-VVpriE_OXkefz7cmxxOu8z_Dumnz4hwe2U0/edit#gid=0)

TL;DR In summary, we find the following to be the best:
- Use a specificly customized preprocessing and augmentation strategy for
each disease category.
- Pretraining a multilabel model on external data and then using this
model for single label classification on some categories improves performance.
- We run each model for a maximum of 5 epochs and average predictions from 
and use a learning rate scheduler that decays based on validation loss and average the predictions from the best two epoch model checkpoints.
- We do the above single model experiment 5 times, each time taking a different 80/20 split of the data
and take the average of the results
- For our final results, we take an ensemble of models primarily consisting of multiple EfficientNets, DenseNets, and InceptionResNets each with their own specific hyperparameter configuration.
## Basic Usage

Python 3.9+
Tensorflow 2.8.0+

You can run an experiment as follows:
```
usage: run_baseline.py [-h] [--epochs EPOCHS] [--dataset_dir DATASET_DIR] [--results_dir RESULTS_DIR] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                       [--num_train_samples NUM_TRAIN_SAMPLES] [--num_test_samples NUM_TEST_SAMPLES] [--train_test_split TRAIN_TEST_SPLIT] [--num_workers NUM_WORKERS] [--freeze_base]
                       [--feature_extraction] [--save_prefix SAVE_PREFIX] [--save_postfix SAVE_POSTFIX] [--custom_save CUSTOM_SAVE] [--log_dir LOG_DIR] [--img_size IMG_SIZE]
                       [--lr_patience LR_PATIENCE] [--data_mode DATA_MODE] [--generator {none,aug}] [--diseases {all,0,1,2,3,4,5,6,7,8,9,10,11,12,13,g1,g2,g3,g4,h2}]
                       [--dataset_type {chexpert,nih}] [--debug_soln_set] [--k K] [--num_folds NUM_FOLDS] [--use_k_folds] [--multi_gpu] --model_type
                       {UNet,Xception,VGG16,VGG19,ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet152V2,InceptionV3,InceptionResNetV2,MobileNetV3Large,MobileNetV3Small,DenseNet121,DenseNet169,DenseNet201,NASNetMobile,NASNetLarge,EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7,EfficientNetV2B0,EfficientNetV2B1,EfficientNetV2B2,EfficientNetV2B3,EfficientNetV2S,EfficientNetV2M,EfficientNetV2L}
                       [--loss_func {default,weighted,weighted-one-class}] [--use_mixed_precision] [--min_rand_nan MIN_RAND_NAN] [--max_rand_nan MAX_RAND_NAN]
                       [--nan_fill_mode {random,zero,one,mean,remove_nan_rows}] [--use_multiprocessing] [--max_queue_size MAX_QUEUE_SIZE] [--load_pretrained LOAD_PRETRAINED] [--remove_top]
                       [--restore_best_weights] [--monitor {val_accuracy,val_loss,val_auc,val_prc}] [--top_config {1,2}] [--print_model_summary] [--label_smoothing LABEL_SMOOTHING]
                       [--not_modify_validation] [--ensemble_best_epochs ENSEMBLE_BEST_EPOCHS] [--lr_reducer_patience LR_REDUCER_PATIENCE] [--double_pop] [--custom_save_dir CUSTOM_SAVE_DIR]
                       [--hp_tuning] [--hp_objective {val_accuracy,val_loss,val_auc,val_prc}] [--hp_max_trials HP_MAX_TRIALS] [--hp_executions_per_trial HP_EXECUTIONS_PER_TRIAL] [--hp_overwrite]
                       [--predict_train_set] [--load_train_set LOAD_TRAIN_SET]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset_dir DATASET_DIR
  --results_dir RESULTS_DIR
  --batch_size BATCH_SIZE
                        Per GPU
  --learning_rate LEARNING_RATE
  --num_train_samples NUM_TRAIN_SAMPLES
  --num_test_samples NUM_TEST_SAMPLES
  --train_test_split TRAIN_TEST_SPLIT
  --num_workers NUM_WORKERS
  --freeze_base
  --feature_extraction
  --save_prefix SAVE_PREFIX
  --save_postfix SAVE_POSTFIX
  --custom_save CUSTOM_SAVE
  --log_dir LOG_DIR
  --img_size IMG_SIZE
  --lr_patience LR_PATIENCE
  --data_mode DATA_MODE
  --generator {none,aug}
  --diseases {all,0,1,2,3,4,5,6,7,8,9,10,11,12,13,g1,g2,g3,g4,h2}
  --dataset_type {chexpert,nih}
  --debug_soln_set
  --k K                 k-fold cross validation; number between 0-4 inclusive
  --num_folds NUM_FOLDS
                        k-fold cross validation
  --use_k_folds
  --multi_gpu           Use multi-gpu training
  --model_type {UNet,Xception,VGG16,VGG19,ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet152V2,InceptionV3,InceptionResNetV2,MobileNetV3Large,MobileNetV3Small,DenseNet121,DenseNet169,DenseNet201,NASNetMobile,NASNetLarge,EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7,EfficientNetV2B0,EfficientNetV2B1,EfficientNetV2B2,EfficientNetV2B3,EfficientNetV2S,EfficientNetV2M,EfficientNetV2L}
  --loss_func {default,weighted,weighted-one-class}
  --use_mixed_precision
  --min_rand_nan MIN_RAND_NAN
  --max_rand_nan MAX_RAND_NAN
  --nan_fill_mode {random,zero,one,mean,remove_nan_rows}
  --use_multiprocessing
  --max_queue_size MAX_QUEUE_SIZE
  --load_pretrained LOAD_PRETRAINED
  --remove_top
  --restore_best_weights
  --monitor {val_accuracy,val_loss,val_auc,val_prc}
  --top_config {1,2}
  --print_model_summary
  --label_smoothing LABEL_SMOOTHING
  --not_modify_validation
  --ensemble_best_epochs ENSEMBLE_BEST_EPOCHS
  --lr_reducer_patience LR_REDUCER_PATIENCE
  --double_pop
  --custom_save_dir CUSTOM_SAVE_DIR
  --hp_tuning
  --hp_objective {val_accuracy,val_loss,val_auc,val_prc}
  --hp_max_trials HP_MAX_TRIALS
  --hp_executions_per_trial HP_EXECUTIONS_PER_TRIAL
  --hp_overwrite
  --predict_train_set
  --load_train_set LOAD_TRAIN_SET
```
## Advanced Usage

We automate the deployment of multiple experiments through [slurm](https://slurm.schedmd.com/).

See `advanced_launcher.py`. It uses configs under subdirectories
under `jobs`.

```
usage: advanced_launcher.py [-h] --config_file CONFIG_FILE [--debug]
                            [--enable_requeue] [--relaunch_failed]
                            [--disease_ids DISEASE_IDS [DISEASE_IDS ...]]
                            [--skip_disease_ids SKIP_DISEASE_IDS [SKIP_DISEASE_IDS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
  --debug
  --enable_requeue
  --relaunch_failed
  --disease_ids DISEASE_IDS [DISEASE_IDS ...]
  --skip_disease_ids SKIP_DISEASE_IDS [SKIP_DISEASE_IDS ...]
```

## License
[GLWTPL](LICENSE)