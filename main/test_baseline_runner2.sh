# python run_baseline.py --batch_size 16 --model_type EfficientNetB0 --epochs 5 --num_workers 1 --save_postfix debug --save_prefix debug \
# --dataset_type nih --dataset_dir /groups/CS156b/teams/ups_data/nih_224

# python run_baseline.py --batch_size 16 --model_type EfficientNetB0 --epochs 5 --num_workers 1 --save_postfix debugn --save_prefix debug \
# --load_pretrained nih_exp_1 --freeze_base --print_model_summary --remove_top --top_config 2

# python run_baseline.py --batch_size 16 --model_type EfficientNetB0 --epochs 5 --num_workers 1 --save_postfix debugn --save_prefix debug \
# --label_smoothing 0.05

# python run_baseline.py --batch_size 16 --model_type EfficientNetB3 --epochs 1 --num_workers 1 --save_postfix debugn --save_prefix debug \
# --monitor val_auc --ensemble_best_epochs 2 \
# --ensemble_best_epochs 1 --custom_save_dir h1_1 --debug_soln_set --diseases h2 \
# --nan_fill_mode zero

# hyperparameter tuning experimen testin
python run_baseline.py --batch_size 16 --model_type EfficientNetB3 --epochs 2 --num_workers 1 \
--monitor val_auc --ensemble_best_epochs 2 \
--custom_save_dir hp_1 --debug_soln_set --diseases 1 \
--nan_fill_mode remove_nan_rows \
--hp_tuning \
--custom_save debug_hypera \
--use_k_folds --k 1
