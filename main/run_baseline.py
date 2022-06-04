import pandas as pd # for some reason this needs to be here for hpc
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import datetime
import argparse
from tqdm import tqdm
import os
import time
import sys
sys.path.append('../')
from helpers import helper
from helpers.loss import weighted_binary_crossentropy, compute_class_weight_one_class
from helpers.preprocessing import get_default_augmentation_image_generator_args
import helpers.models as models_helper
import numpy as np
from loguru import logger
from icecream import ic
import json
import keras_tuner

startTime = time.time()
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tqdm.pandas()

parser = argparse.ArgumentParser()
helper.add_base_parser_args(parser)
parser.add_argument('--multi_gpu', action='store_true', help='Use multi-gpu training', default=False)
parser.add_argument('--model_type', type=str, required=True, choices=models_helper.get_models())
parser.add_argument('--loss_func', type=str, default="default", choices=["default", "weighted", "weighted-one-class"])
parser.add_argument('--use_mixed_precision', action='store_true', default=False)

parser.add_argument('--min_rand_nan', type=float, default=-1.0)
parser.add_argument('--max_rand_nan', type=float, default=1.0)
parser.add_argument('--nan_fill_mode', type=str, default="mean", choices=["random", "zero", "one", "mean", "remove_nan_rows"])

parser.add_argument('--use_multiprocessing', action='store_true', default=False)
parser.add_argument('--max_queue_size', type=int, default=10)
parser.add_argument('--load_pretrained', type=str, default=None)
parser.add_argument('--remove_top', default=False, action='store_true')

parser.add_argument('--restore_best_weights', action='store_true', default=False)
parser.add_argument('--monitor', type=str, default="val_loss", choices=["val_accuracy", "val_loss", "val_auc", "val_prc"])
parser.add_argument('--top_config', type=str, default="1", choices=["1", "2"])

parser.add_argument('--print_model_summary', action='store_true', default=False)

parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--not_modify_validation', action='store_true', default=False)

parser.add_argument('--ensemble_best_epochs', type=int, default=2)

parser.add_argument('--lr_reducer_patience', type=int, default=10)

parser.add_argument('--double_pop', action='store_true', default=False)

parser.add_argument('--custom_save_dir', type=str, default=None)

parser.add_argument('--hp_tuning', action='store_true', default=False)
parser.add_argument('--hp_objective', type=str, default="val_loss", choices=["val_accuracy", "val_loss", "val_auc", "val_prc"])
parser.add_argument('--hp_max_trials', type=int, default=10)
parser.add_argument('--hp_executions_per_trial', type=int, default=1)
parser.add_argument('--hp_overwrite', action='store_true', default=False)

args = parser.parse_args()

job_id = os.environ.get('SLURM_JOB_ID') or "no_job_id"

logger.info(f"Starting job {job_id}")

logger.success(tf.config.list_physical_devices('GPU'))

hp_learning_rate_choices = [1e-3, 1e-4, 3e-4, 1e-5, 6e-4]

if args.multi_gpu:
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    logger.info('Using Multi-GPU Mirrored Strategy: {}'.format(strategy.num_replicas_in_sync))
    assert(num_gpus > 1)
else:
    logger.info("Using Single-GPU (No Mirrored)")

if not args.img_size:
    args.img_size = models_helper.get_model_default_img_size(args.model_type)
    logger.info(f"Using model's default image input size: {args.img_size}")
else:
    logger.info(f"Using custom image size: {args.img_size}")

logger.debug(args)

if args.use_mixed_precision:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    logger.info("Using mixed precision")
model_type = args.model_type

h = helper.DataHelper(args)

def handle_custom_save_dir(output_file_name):
    if args.custom_save_dir:
        output_file_name = os.path.join(args.custom_save_dir, output_file_name)
        os.makedirs(h.models_dir + "/" + args.custom_save_dir, exist_ok=True)
        os.makedirs(h.results_dir + "/" + args.custom_save_dir, exist_ok=True)
        os.makedirs('../logs' + "/" + args.custom_save_dir, exist_ok=True)
    return output_file_name

if args.custom_save:
    output_file_name = args.custom_save
    output_file_name = handle_custom_save_dir(output_file_name)
    logger.info(f"Custom Output File: {output_file_name}")
else:
    job_id = os.environ.get('SLURM_JOB_ID') or "no_job_id"
    output_file_name = f"{args.save_prefix}_{model_type}_{job_id}_{args.save_postfix}"
    output_file_name = handle_custom_save_dir(output_file_name)
    logger.info(f"Output File: {output_file_name}")

def get_loss_fn():
    if args.loss_func == "default" or "weighted-one-class":
        return tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
    elif args.loss_func == "weighted":
        return weighted_binary_crossentropy(h)
    else:
        assert False, f"Unknown loss func: {args.loss_func}"

custom_loss = get_loss_fn()

# compute class weight before preprocessing labels
if args.loss_func == "weighted-one-class":
    class_weight = compute_class_weight_one_class(h.train_df, h.diseases)

if args.nan_fill_mode == "remove_nan_rows":
    assert args.diseases != 'all', "Can't remove nan rows if using all diseases"
    h.remove_nan_rows()
elif args.nan_fill_mode == "mean":
    h.fill_dataset_nan_with_mean(modify_validation=not args.not_modify_validation)
    h.fill_dataset_nan(with_val=0, modify_validation=not args.not_modify_validation) # backup, only for small training dataset when some cols are all nan
elif args.nan_fill_mode == "random":
    h.fill_dataset_nan_with_random(args.min_rand_nan, args.max_rand_nan, modify_validation=not args.not_modify_validation)
elif args.nan_fill_mode == "zero":
    h.fill_dataset_nan(with_val=-1, modify_validation=not args.not_modify_validation) # -1 is zero
elif args.nan_fill_mode == "one":
    h.fill_dataset_nan(with_val=1, modify_validation=not args.not_modify_validation)

if args.not_modify_validation:
    assert args.monitor != "val_loss", "Can't use val_loss monitor if not modifying validation"

if args.dataset_type == "chexpert": 
    h.shift_train_dataset_to_0_1_range()

logger.debug(h.train_df.head())

def build_image_generators(model_type):

    preprocessing_fn = models_helper.get_model_default_preprocessing(model_type)
    
    valid_test_generator_args = {"preprocessing_function": preprocessing_fn}
    if args.generator == "aug":
        aug_args = get_default_augmentation_image_generator_args()
    else:
        aug_args = {}
    aug_args["preprocessing_function"] = preprocessing_fn 

    return h.get_image_generators(img_target_size=(args.img_size, args.img_size), augmentation_args=aug_args,
                                  valid_test_image_generator_args=valid_test_generator_args)

train_gen, valid_gen, test_gen, soln_gen = build_image_generators(model_type)

def build_model(hp):
    if args.load_pretrained:
        model = models_helper.load_model(args.load_pretrained, remove_top=args.remove_top, top_config=args.top_config, freeze_base=args.freeze_base, num_classes=len(h.diseases))
        logger.info(f"Loaded pretrained model: {args.load_pretrained}")
    else:
        model = models_helper.get_model(model_type, len(h.diseases), img_size=args.img_size, freeze_base=args.freeze_base, feature_extraction=args.feature_extraction, double_pop=args.double_pop)

    METRICS = [
        tf.keras.metrics.AUC(name='auc', multi_label=True),
        tf.keras.metrics.AUC(name='prc', curve='PR', multi_label=True),  # precision-recall curve
        tf.keras.metrics.MeanSquaredError(name='mse'),
        'accuracy',
    ]
    if hp:
        learning_rate = hp.Choice('learning_rate', hp_learning_rate_choices)
    else:
        learning_rate = args.learning_rate
    
    model.compile(
        loss=custom_loss,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=METRICS
    )

    if args.print_model_summary:
        model.summary()

    return model

if not args.hp_tuning:
    hp = None
else:
    hp = keras_tuner.HyperParameters()

model_type = args.model_type
if args.multi_gpu:
    with strategy.scope():
        model = build_model(hp)
else:
    model = build_model(hp)

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=args.log_dir + '/' + output_file_name),
    tf.keras.callbacks.EarlyStopping(patience=args.lr_patience, restore_best_weights=args.restore_best_weights, monitor=args.monitor),
    tf.keras.callbacks.ReduceLROnPlateau(monitor=args.monitor, patience=args.lr_reducer_patience),
    tf.keras.callbacks.ModelCheckpoint(
        f"{h.models_dir}/{output_file_name}.h5",
        save_best_only=True
    ),
    tf.keras.callbacks.CSVLogger(f"{h.results_dir}/{output_file_name}_LOG.csv")
]

if args.ensemble_best_epochs:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        f"{h.models_dir}/{output_file_name}_ckpts/" + "ckpt_{epoch:02d}.h5",
        monitor=args.monitor,
        save_weights_only=True,
    ))
    os.makedirs(f"{h.models_dir}/{output_file_name}_ckpts", exist_ok=True)

fit_arguments = {
    "epochs": args.epochs,
    "validation_data": valid_gen,
    "workers": args.num_workers,
    "use_multiprocessing": args.use_multiprocessing,
    "callbacks": callbacks,
    "max_queue_size": args.max_queue_size,
}

if args.loss_func == "weighted-one-class":
    fit_arguments["class_weight"] = class_weight

if args.hp_tuning:
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective=args.hp_objective,
        max_trials=args.hp_max_trials,
        executions_per_trial=args.hp_executions_per_trial,
        directory=f"{h.models_dir}/{output_file_name}_hp_tuning",
        project_name=output_file_name,
        overwrite=args.hp_overwrite,
    )
    tuner.search(train_gen, **fit_arguments)
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    model = build_model(best_hps[0])

hist = model.fit(
    train_gen,
    **fit_arguments
)

logger.info("Predicting test set...")
test_y = model.predict(test_gen, workers=args.num_workers, use_multiprocessing=args.use_multiprocessing)
test_y = h.shift_values_to_normal_range(test_y)
h.save_preds(test_y, output_file_name, mode="test")

logger.info("Predicting validation set...")
valid_y = model.predict(valid_gen, workers=args.num_workers, use_multiprocessing=args.use_multiprocessing)
valid_y = h.shift_values_to_normal_range(valid_y)
h.save_preds(valid_y, output_file_name, mode="valid")

logger.info("Predicting solution set...")
soln_y = model.predict(soln_gen, workers=args.num_workers, use_multiprocessing=args.use_multiprocessing)
soln_y = h.shift_values_to_normal_range(soln_y)
h.save_preds(soln_y, output_file_name, mode="soln")

if args.ensemble_best_epochs:
    logger.info("Ensemble best epochs...")
    epochs_metrics = list(hist.history[args.monitor])
    best_metrics = epochs_metrics.copy()
    best_metrics.sort()
    if "loss" in args.monitor:
        best_metrics.reverse()
    test_preds = None
    soln_preds = None
    for metric in best_metrics[-args.ensemble_best_epochs:]:
        epoch = epochs_metrics.index(metric) + 1
        model.load_weights(f"{h.models_dir}/{output_file_name}_ckpts/ckpt_{epoch:02d}.h5")
        test_y = model.predict(test_gen, workers=args.num_workers, use_multiprocessing=args.use_multiprocessing)
        soln_y = model.predict(soln_gen, workers=args.num_workers, use_multiprocessing=args.use_multiprocessing)
        test_y = h.shift_values_to_normal_range(test_y)
        soln_y = h.shift_values_to_normal_range(soln_y)
        if test_preds is None:
            test_preds = test_y
            soln_preds = soln_y
        else:
            test_preds += test_y
            soln_preds += soln_y
    test_preds /= args.ensemble_best_epochs
    soln_preds /= args.ensemble_best_epochs
    h.save_preds(test_preds, output_file_name, mode="test", ensemble=True)
    h.save_preds(soln_preds, output_file_name, mode="soln", ensemble=True)

executionTime = (time.time() - startTime)
args.execution_time = str(executionTime) + " seconds"
job_id = os.environ.get('SLURM_JOB_ID')
args.job_id = job_id
args.val_loss = hist.history['val_loss']
args.val_accuracy = hist.history['val_accuracy']
args.val_auc = hist.history['val_auc']
args.current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.info(f"Validation loss: {hist.history['val_loss']}")
logger.info(f"Validation accuracy: {hist.history['val_accuracy']}")
h.save_config_args(output_file_name, args)
logger.info("Done")
