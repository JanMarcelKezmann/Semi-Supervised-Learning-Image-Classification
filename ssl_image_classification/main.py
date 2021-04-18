import argparse
import os
from absl import logging
from libml.utils import setup_tf, load_config

setup_tf()

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm

from libml.models import get_model
from libml.optimizers import get_optimizer
from libml.preprocess import fetch_dataset
from libml.train_utils import ema, weight_decay, linear_rampup
from libml.data_augmentations import weak_augment, medium_augment, strong_augment

from algorithms import mixup, mixmatch, remixmatch, fixmatch, vat, meanteacher, pimodel, pseudolabel, ict


tfd = tfp.distributions


def get_args(parser_args=[]):
    """
    Define Argument Parser.

    Args:
        parser_args: list, taking argument and corresponding value as elements
                        (Only necessary in Jupyter Notebooks or similar)

    Returns:
        dictionary containing arguments of parser as keys and its corresponding 
        values as values.

    Example:
        In case you are working in an Jupyter Notebook environment like Google
        Colaboratory it makes sense to givr the main() function a list containing
        all non default arguments for the argument parser.
        This means, if you e.g. would like to change the config-path and set the
        notebook setting to True create a list like this:

        parser_args = ["--epochs", "2", "--config-path", "dataset configurations", "--notebook"]

        Name the argument first and the corresponding value second.
        In case of Boolean arguments only the argument itself is necessary.
    """
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--seed", type=int, default=[1, 2], help="Seed for repeatable results.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"], help="Dataset for training")
    parser.add_argument("--algorithm", type=str, default="ict", choices=["mixup", "mixmatch", "remixmatch", "fixmatch", "vat", "mean teacher", "pseudo label", "pi-model", "ict"], help="Semi-Supervised Learning Method")
    parser.add_argument("--model", type=str, default="efficientnetb0", help="CNN Model for Classification")
    parser.add_argument("--weights", type=str, default=None, choices=[None, "imagenet"], help="Initial Weights of the Model")

    # Training related arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of Epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size")
    parser.add_argument("--pre-val-iter", type=int, default=100, help="Number of Iterations previous to Validation")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Exponential Moving Average Decay for EMA Model")
    
    # Optimizer related arguments
    parser.add_argument("--optimizer", type=str, default="SGD", choices=['AdaDelta', 'AdaGrad', 'Adam', 'Adamax', 'Nadam', 'RMSProp', 'SGD'], help="Optimizer of the model")
    parser.add_argument("--lr", type=int, default=1e-1, help="Learning Rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 parameter for various optimizers")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 parameter for various optimizers")
    parser.add_argument("--rho", type=float, default=0.9, help="Rho parameter for various optimizers")

    # Loss related arguments
    parser.add_argument("--lambda-u", type=int, default=10, help="Unlabeled Loss Multiplier used for almost all algorithms")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight Decay Rate")
    parser.add_argument("--threshold", type=float, default=0.95, help="Confidence or threshold parameter used in multiple unsupervised losses (Fixmatch and Pseudo label)")

    # Dataset related arguments
    parser.add_argument("--num-lab-samples", type=int, default=4000, help="Total Number of labeled samples")
    parser.add_argument("--val-samples", type=int, default=1000, help="Total Number of validation samples")
    parser.add_argument("--total-train-samples", type=int, default=50000, help="Total number of train samples")
    parser.add_argument("--height", type=int, default=32, help="Input Height of Image")
    parser.add_argument("--width", type=int, default=32, help="Input Width of Image")

    # Algorithm related arguments
    # Additional arugments for MixUp, MixMatch, ReMixMatch and ICT
    parser.add_argument("--alpha", type=float, default=0.75, help="Beta Distribution parameter")

    # Additional arguments for Mixmatch, ReMixMatch
    parser.add_argument("--T", type=float, default=0.5, help="Temperature sharpening ratio")
    parser.add_argument("--K", type=int, default=2, help="Amount of augmentation rounds")
    
    # Additional arguments for ReMixMatch
    parser.add_argument("--w-rot", type=float, default=0.5, help="Rotation loss multiplier")
    parser.add_argument("--w-kl", type=float, default=0.5, help="KL loss multiplier")
    
    # Additional Arguments for VAT
    parser.add_argument("--vat-eps", type=float, default=6, help="VAT perturbation size")
    parser.add_argument("--w-entropy", type=float, default=0.06, help="Entropy loss weight")

    # Further Arguments
    parser.add_argument("--config-path", type=str, default=None, help="Path to YAML config file (Overwrite args)")
    parser.add_argument("--tensorboard", action="store_true", help="TensorBoard Visualization Enabler")
    parser.add_argument("--resume", action="store_true", help="Bool for restoring from preious training runs")
    parser.add_argument("--notebook", action="store_true", help="Bool for TQDM training visualization")

    return parser.parse_args(args=parser_args)
    # return parser.parse_args()


def main():
    """
    Main function that loads configurations, fetches data, defines the model,
    optimizer and further classes, runs training, validation and testing and
    saves every result.

    Args:
        parser_args: list, contains keys and values for argument parser, see 
        Example in get_args() for more information

    Returns:
        None
    """
    # Set __dict__ attribute of get_args()
    args = vars(get_args())
    # # Get directory name of real path
    # dir_path = os.path.dirname(os.path.realpath(__file__))

    if args["config_path"] is not None and os.path.exists(args["config_path"]):
        args = load_config(args)
    print(args)

    start_epoch = 0
    best_val_accuracy = 0.0
    log_path = f"logs/{args['dataset']}@{args['num_lab_samples']}"
    ckpt_dir = f"{log_path}/checkpoints"

    labeled_data, unlabeled_data, val_data, test_data, num_classes = fetch_dataset(args, log_path)

    # Define Model, Optimizer and Checkpoints
    model = get_model(name=args["model"], weights=args["weights"], height=args["height"], width=args["width"], classes=num_classes)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args["lr"],
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )
    optimizer = get_optimizer(opt_name=args["optimizer"], lr=args["lr"], momentum=args["momentum"], beta1=args["beta1"], beta2=args["beta2"], rho=args["rho"])
    model_ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(model_ckpt, f"{ckpt_dir}/model", max_to_keep=10)

    # Define EMA-Model, EMA-Model Weights and EMA-Checkpoint
    ema_model = get_model(name=args["model"], weights=args["weights"], height=args["height"], width=args["width"], classes=num_classes)
    ema_model.set_weights(model.get_weights())
    ema_model_ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=ema_model)
    ema_manager = tf.train.CheckpointManager(ema_model_ckpt, f"{ckpt_dir}/ema_model", max_to_keep=5)

    # restore previous checkpoints if exist for model and ema_model including last epoch
    if args["resume"]:
        model_ckpt.restore(manager.latest_checkpoint)
        ema_model_ckpt.restore(manager.latest_checkpoint)
        model_ckpt.step.assign_add(1)
        ema_model_ckpt.step.assign_add(1)
        start_epoch = int(model_ckpt.step)
        print(f"Restored @ epoch {start_epoch} from {manager.latest_checkpoint} and {ema_manager.latest_checkpoint}")

    # Create summary file writer for given log directory for training, validation and testing
    train_writer = None
    if args["tensorboard"]:
        train_writer = tf.summary.create_file_writer(f"{log_path}/train")
        val_writer = tf.summary.create_file_writer(f"{log_path}/val")
        test_writer = tf.summary.create_file_writer(f"{log_path}/test")

    # Assigning args used in functions wrapped with tf.function to tf.constant/tf.Variable to avoid memory leaks
    args["T"] = tf.constant(args["T"])
    args["K"] = tf.constant(args["K"])
    args["epochs"] = tf.constant(args["epochs"])
    args["pre_val_iter"] = tf.constant(args["pre_val_iter"])
    args["threshold"] = tf.constant(args["threshold"])
    args["height"] = tf.constant(args["height"])
    args["width"] = tf.constant(args["width"])
    args["alpha"] = tf.constant(args["alpha"])
    args["w_rot"] = tf.constant(args["w_rot"])
    args["w_kl"] = tf.constant(args["w_kl"])
    args["vat_eps"] = tf.constant(args["vat_eps"])
    if args["algorithm"].lower() == "mixmatch" or args["algorithm"].lower() == "remixmatch":
        args["beta"] = tf.Variable(0., shape=())
    elif args["algorithm"].lower() == "mixup" or args["algorithm"].lower() == "ict":
        args["beta"] = tf.Variable(tf.zeros(shape=(args["batch_size"], 1, 1, 1), dtype=tf.dtypes.float32), shape=(args["batch_size"], 1, 1, 1))
    
    # Loop over all (remaining) epochs
    for epoch in range(start_epoch, args["epochs"]):
        x_loss, u_loss, total_loss, accuracy = train(labeled_data, unlabeled_data, model, ema_model, optimizer, epoch, num_classes, args)

        val_x_loss, val_accuracy = validate(val_data, ema_model, epoch, args, split="Validation")
        test_x_loss, test_accuracy = validate(test_data, ema_model, epoch, args, split="Test")

        if (epoch - start_epoch) % 10 == 0:
            model_save_path = manager.save(checkpoint_number=int(model_ckpt.step))
            ema_model_save_path = ema_manager.save(checkpoint_number=int(ema_model_ckpt.step))
            print(f"Saved Model checkpoint for epoch {int(model_ckpt.step)} @ {model_save_path}")
            print(f"Saved EMA-Model checkpoint for epoch {int(ema_model_ckpt.step)} @ {ema_model_save_path}")

        # Update Model/EMA-Model checkpoint step
        model_ckpt.step.assign_add(1)
        ema_model_ckpt.step.assign_add(1)

        # Update log writer for Tensorboard
        step = args["pre_val_iter"] * (epoch + 1)
        if args["tensorboard"]:
            with train_writer.as_default():
                tf.summary.scalar("x_loss", x_loss.result(), step=step)
                tf.summary.scalar("u_loss", u_loss.result(), step=step)
                tf.summary.scalar("total_loss", total_loss.result(), step=step)
                tf.summary.scalar("accuracy", accuracy.result(), step=step)
            with val_writer.as_default():
                tf.summary.scalar("x_loss", val_x_loss.result(), step=step)
                tf.summary.scalar("accuracy", val_accuracy.result(), step=step)
            with test_writer.as_default():
                tf.summary.scalar("x_loss", test_x_loss.result(), step=step)
                tf.summary.scalar("accuracy", test_accuracy.result(), step=step)

    # Send buffered data of summary writer (for train/val/test) to storage
    if args["tensorboard"]:
        for writer in [train_writer, val_writer, test_writer]:
            writer.flush()

def train(labeled_data, unlabeled_data, model, ema_model, opt, epoch, num_classes, args):
    x_loss_avg = tf.keras.metrics.Mean()
    u_loss_avg = tf.keras.metrics.Mean()
    l2_loss_avg = tf.keras.metrics.Mean()
    rot_loss_avg = tf.keras.metrics.Mean()
    kl_loss_avg = tf.keras.metrics.Mean()
    entropy_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    acc = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args["batch_size"], drop_remainder=True)

    if args["algorithm"] == "fixmatch":
        uratio = int(np.ceil(int(len(unlabeled_data)) / args["num_lab_samples"]))
        if uratio / 2 % 2 != 0:
            uratio = (uratio // 2) * 2

        # update pre_val_iter such that every sample will be used only ones per epoch (especially unlabeled ones)
        args["pre_val_iter"] = int(np.floor(args["num_lab_samples"] / args["batch_size"]))
    else:
        uratio = 1
    shuffle_and_batch_unlabeled = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args["batch_size"] * uratio, drop_remainder=True)
    
    # Define iterator that holds args["batch_size"] amount of images
    iter_labeled_data = iter(shuffle_and_batch(labeled_data))
    iter_unlabeled_data = iter(shuffle_and_batch_unlabeled(unlabeled_data))

    if args["notebook"]:
        prog_bar = tqdm.notebook.tqdm(range(args["pre_val_iter"]), unit="batch")
    else:
        prog_bar = tqdm.tqdm(range(args["pre_val_iter"]), unit="batch")
    
    for batch_num in prog_bar:
        # Get next batch of labeled and unlabeled data
        try:
            labeled_batch = next(iter_labeled_data)
        except:
            iter_labeled_data = iter(shuffle_and_batch(labeled_data))
            labeled_batch = next(iter_labeled_data)
        
        try:
            unlabeled_batch = next(iter_unlabeled_data)
        except:
            iter_unlabeled_data = iter(shuffle_and_batch(unlabeled_data))
            unlabeled_batch = next(iter_unlabeled_data)

        with tf.GradientTape() as tape:
            # run SSL Algorithm of choice
            ###
            # needs to be change later on to be more general
            ###            
            if args["algorithm"].lower() == "mixup":
                # Set Beta distribution parameters in args
                # args["beta"].assign(tf.compat.v1.distributions.Beta(args["alpha"], args["alpha"]).sample([args["batch_size"], 1, 1, 1]))
                args["beta"].assign(tfp.distributions.Beta(args["alpha"], args["alpha"]).sample([args["batch_size"], 1, 1, 1]))

                # Run Mixup
                X, labels_X = mixup(
                    labeled_batch["image"],
                    labeled_batch["image"][::-1],
                    labeled_batch["label"],
                    labeled_batch["label"][::-1],
                    args["beta"],
                    "mixup"
                    )
                
                # Get Model predictions
                logits_X = model(X, training=True)[0]

                # Run Mixup and get labels
                U, labels_U = mixup(
                    unlabeled_batch["image"],
                    unlabeled_batch["image"][::-1],
                    tf.nn.softmax(model(unlabeled_batch["image"], training=True)[0], axis=1),
                    tf.nn.softmax(model(unlabeled_batch["image"], training=True)[0], axis=1)[::-1],
                    args["beta"],
                    "mixup"
                    )
                
                labels_U = tf.stop_gradient(labels_U)
                # Compute Model Predictions
                logits_U = model(U, training=True)[0]

                # Compute Loss
                x_loss, u_loss = ssl_loss_mixup(labels_X, logits_X, labels_U, logits_U)
                total_loss = x_loss + u_loss

            elif args["algorithm"].lower() == "mixmatch":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(args["epochs"], epoch, args["pre_val_iter"], iteration)

                # Set Beta distribution parameters in args
                # args["beta"].assign(np.random.beta(args["alpha"], args["alpha"]))
                args["beta"].assign(tfp.distributions.Beta(args["alpha"], args["alpha"]).sample([1])[0])

                X_prime, U_prime = mixmatch(
                    model,
                    labeled_batch["image"],
                    labeled_batch["label"],
                    unlabeled_batch["image"],
                    args["T"],
                    args["K"],
                    args["beta"],
                    args["height"],
                    args["width"]
                    )
                
                # Get model predictions
                logits = [model(X_prime[0], training=True)[0]]
                for batch in X_prime[1:]:
                    logits.append(model(batch, training=True)[0])
                logits = interleave(logits, args["batch_size"])
                logits_X = logits[0]
                logits_U = tf.concat(logits[1:], axis=0)
                
                # Compute supervised and unsupervised losses
                x_loss, u_loss = ssl_loss_mixmatch(U_prime[:args["batch_size"]], logits_X, U_prime[args["batch_size"]:], logits_U)
                # Compute l2 loss
                wd_loss = sum(tf.nn.l2_loss(v) for v in model.trainable_variables if "predictions" in v.name) # maybe run tf.nn.softmax(v)
                
                total_loss = x_loss + lambda_u * u_loss + args["wd"] * wd_loss


            elif args["algorithm"].lower() == "remixmatch":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(args["epochs"], epoch, args["pre_val_iter"], iteration)

                # Set Beta distribution parameters in args
                # args["beta"].assign(np.random.beta(args["alpha"], args["alpha"]))
                args["beta"].assign(tfp.distributions.Beta(args["alpha"], args["alpha"]).sample([1])[0])

                rot_loss = compute_rot_loss(weak_augment(unlabeled_batch["image"], args["height"], args["width"]), model, w_rot=args["w_rot"])

                X_prime, U_prime, kl_loss = remixmatch(
                    model,
                    labeled_batch["image"], # xt_in
                    labeled_batch["label"], # l_in
                    unlabeled_batch["image"], # y_in
                    args["T"],
                    args["K"],
                    args["beta"],
                    args["height"],
                    args["width"]
                    )

                # Get model predictios                
                logits = [model(batch, training=True)[0] for batch in X_prime[:-1]]
                logits.append(model(X_prime[-1], training=True)[0])
                logits = interleave(logits, args["batch_size"])
                logits_X = logits[0]
                logits_U = tf.concat(logits[1:], axis=0)
                
                # Compute labeled and unlabeled loss
                x_loss, u_loss = ssl_loss_remixmatch(labeled_batch["label"], logits_X, U_prime[args["batch_size"]:], logits_U)                
                # Compute L2 loss
                wd_loss = sum(tf.nn.l2_loss(v) for v in model.trainable_variables if "predictions" in v.name)
                
                total_loss = x_loss + lambda_u * u_loss + args["wd"] * wd_loss + args["w_rot"] * rot_loss + args["w_kl"] * kl_loss

            elif args["algorithm"].lower() == "fixmatch":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(args["epochs"], epoch, args["pre_val_iter"], iteration)

                x_aug, labels_strong, u_strong_aug = fixmatch(
                    model,
                    labeled_batch["image"], # xt_in
                    labeled_batch["label"], # l_in
                    unlabeled_batch["image"], # y_in
                    args["height"],
                    args["width"],
                    uratio=uratio
                )

                # Get model predictions
                logits = [model(x_aug, training=True)[0]]
                for i in range(uratio):
                    logits.append(model(u_strong_aug[i], training=True)[0])
                logits_x = logits[0]
                logits_strong = tf.concat(logits[1:], axis=0) # shape = (uratio * batch, num_classes)

                # Compute supervised and unsupervised loss
                x_loss, u_loss = ssl_loss_fixmatch(labeled_batch["label"], logits_x, labels_strong, logits_strong, args["threshold"])
                # Compute L2 loss
                wd_loss = sum(tf.nn.l2_loss(v) for v in model.trainable_variables if "predictions" in v.name)
                
                total_loss = x_loss + lambda_u * u_loss + args["wd"] * wd_loss

            elif args["algorithm"].lower() == "vat":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(args["epochs"], epoch, args["pre_val_iter"], iteration)

                # Compute model outputs
                logits_x = model(labeled_batch["image"], training=True)[0]
                logits_u = model(unlabeled_batch["image"], training=True)[0]

                delta_u = vat(
                    unlabeled_batch["image"],
                    logits_u,
                    model,
                    tf.random.Generator.from_non_deterministic_state(),
                    args["vat_eps"]
                    )

                logits_student = model(unlabeled_batch["image"] + delta_u, training=True)[0]
                logits_teacher = tf.stop_gradient(logits_u)

                # Compute supervised and unsupervised loss and unsupervised shannon entropy
                x_loss, u_loss, loss_entropy = ssl_loss_vat(labeled_batch["label"], logits_x, logits_student, logits_teacher, logits_u)

                total_loss = x_loss + lambda_u * u_loss + args["w_entropy"] * loss_entropy
            
            elif args["algorithm"].lower() == "mean teacher":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(args["epochs"], epoch, args["pre_val_iter"], iteration)

                x_augment, u_teacher, u_student = mean_teacher(
                    labeled_batch["image"],
                    unlabeled_batch["image"],
                    args["height"],
                    args["width"]
                )

                # Compute model outputs
                logits_x = model(x_augment, training=True)[0]

                logits_teacher = ema_model(u_teacher, training=True)[0]
                logits_teacher = tf.stop_gradient(logits_teacher)
                logits_student = model(u_student, training=True)[0]

                # Compute supervised and unsupervised losses
                x_loss, u_loss = ssl_loss_mean_teacher(labeled_batch["label"], logits_x, logits_teacher, logits_student)
                # L2 regularization
                wd_loss = sum(tf.nn.l2_loss(v) for v in model.trainable_variables if "predictions" in v.name)
                
                total_loss = x_loss + lambda_u * u_loss + args["wd"] * wd_loss
            
            elif args["algorithm"].lower() == "pseudo label":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(args["epochs"], epoch, args["pre_val_iter"], iteration)
                
                x_augment, u_augment = pseudo_label(
                    labeled_batch["image"],
                    unlabeled_batch["image"],
                    args["height"],
                    args["width"]
                )

                # Get Model outputs
                logits_x = model(x_augment, training=True)[0]
                logits_u = model(u_augment, training=True)[0]

                # Compute supervised and unsupervised losses
                x_loss, u_loss = ssl_loss_pseudo_label(labeled_batch["label"], logits_x, logits_u, args["threshold"])
                # L2 regularization
                wd_loss = sum(tf.nn.l2_loss(v) for v in model.trainable_variables if "predictions" in v.name)
                
                total_loss = x_loss + lambda_u * u_loss + args["wd"] * wd_loss
            
            elif args["algorithm"].lower() == "pi-model":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(args["epochs"], epoch, args["pre_val_iter"], iteration)

                x_augment, u_teacher, u_student = pi_model(
                    labeled_batch["image"],
                    unlabeled_batch["image"],
                    args["height"],
                    args["width"]
                )

                # Computer model outputs
                logits_x = model(x_augment, training=True)[0]
                logits_teacher = model(u_teacher, training=True)[0]
                logits_teacher = tf.stop_gradient(logits_teacher)
                logits_student = model(u_student, training=True)[0]

                # Compute supervised and unsupervised losses
                x_loss, u_loss = ssl_loss_pi_model(labeled_batch["label"], logits_x, logits_teacher, logits_student)
                # L2 regularization
                wd_loss = sum(tf.nn.l2_loss(v) for v in model.trainable_variables if "predictions" in v.name)

                total_loss = x_loss + lambda_u * u_loss + args["wd"] * wd_loss

            elif args["algorithm"].lower() == "ict":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(args["epochs"], epoch, args["pre_val_iter"], iteration)
                # Set Beta distribution parameters in args
                args["beta"].assign(tfp.distributions.Beta(args["alpha"], args["alpha"]).sample([args["batch_size"], 1, 1, 1]))

                x_augment, u_teacher, u_student = ict(
                    labeled_batch["image"],
                    unlabeled_batch["image"],
                    args["height"],
                    args["width"]
                )

                # Get model outputs and labels
                logits_x = model(x_augment, training=True)[0]
                ema_logits_teacher = ema_model(u_teacher, training=True)[0]
                ema_labels_teacher = tf.stop_gradient(tf.nn.softmax(ema_logits_teacher))
                ema_logits_student = ema_model(u_student, training=True)[0]
                ema_labels_student = tf.stop_gradient(tf.nn.softmax(ema_logits_student))
                
                u_student, labels_teacher = mixup(
                    u_teacher,
                    u_student,
                    ema_labels_teacher,
                    ema_labels_student,
                    args["beta"],
                    "mixup"
                    )

                # Get model outputs
                logits_student = model(u_student, training=True)[0]

                # Compute supervised and unsupervised losses
                x_loss, u_loss = ssl_loss_ict(labeled_batch["label"], logits_x, labels_teacher, logits_student)
                
                total_loss = x_loss + lambda_u * u_loss
            else:
                raise ValueError("The argument 'algorithm' (in args['algorithm']) in the argument parser must be one of 'mixup', 'mixmatch' or 'remixmatch'. ")

        # Compute Gradients
        grads = tape.gradient(total_loss, model.trainable_variables)
        ###
        # Update learning rate currently done with a lr scheduler while initializing optimizer
        # opt.learning_rate = 
        ###
        # Run Training Step
        opt.apply_gradients(zip(grads, model.trainable_variables))

        # Update Exponential Moving Average and Weight Decay
        ema(model, ema_model, args["ema_decay"])
        weight_decay(model=model, decay_rate=args["wd"] * args["lr"])

        # Update average losses and accuracy
        x_loss_avg(x_loss)
        u_loss_avg(u_loss)
        total_loss_avg(total_loss)
        acc(tf.argmax(labeled_batch["label"], axis=1, output_type=tf.int32), model(tf.cast(labeled_batch["image"], dtype=tf.float32), training=False)[0])

        # Update Progress Bar and additional losses
        if args["algorithm"].lower() in ["mixup"]:
            prog_bar.set_postfix(
                {
                "X Loss": f"{x_loss_avg.result():.4f}",
                "U Loss": f"{u_loss_avg.result():.4f}",
                "Total Loss": f"{total_loss_avg.result():.4f}",
                "Accuracy": f"{acc.result():.3%}"
                }
                )

        elif args["algorithm"].lower() in ["ict"]:
            prog_bar.set_postfix(
                {
                "X Loss": f"{x_loss_avg.result():.4f}",
                "U Loss": f"{u_loss_avg.result():.4f}",
                "Lambda-U": f"{lambda_u:.3f}",
                "Total Loss": f"{total_loss_avg.result():.4f}",
                "Accuracy": f"{acc.result():.3%}"
                }
                )
                        
        elif args["algorithm"].lower() in ["mixmatch", "fixmatch", "mean teacher", "pseudo label", "pi-model"]:
            l2_loss_avg(wd_loss)

            prog_bar.set_postfix(
                {
                "X Loss": f"{x_loss_avg.result():.4f}",
                "U Loss": f"{u_loss_avg.result():.4f}",
                "Lambda-U": f"{lambda_u:.3f}",
                "Weighted L2 Loss": f"{args['wd'] * l2_loss_avg.result():.4f}",
                "Total Loss": f"{total_loss_avg.result():.4f}",
                "Accuracy": f"{acc.result():.3%}"
                }
                )
                        
        elif args["algorithm"].lower() == "remixmatch":
            l2_loss_avg(wd_loss)
            rot_loss_avg(rot_loss)
            kl_loss_avg(kl_loss)

            prog_bar.set_postfix(
                {
                    "X Loss": f"{x_loss_avg.result():.4f}",
                    "U Loss": f"{u_loss_avg.result():.4f}",
                    "Lambda-U": f"{lambda_u:.3f}",
                    "Weighted L2 Loss": f"{args['wd'] * l2_loss_avg.result():.4f}",
                    "Weighted Rotation Loss": f"{args['w_rot'] * rot_loss_avg.result():.4f}",
                    "Weighted KL Loss": f"{args['w_kl'] * kl_loss_avg.result():.4f}",
                    "Total Loss": f"{total_loss_avg.result():.4f}",
                    "Accuracy": f"{acc.result():.3%}"
                }
                )
        
        elif args["algorithm"].lower() == "vat":
            entropy_loss_avg(loss_entropy)

            prog_bar.set_postfix(
                {
                    "X Loss": f"{x_loss_avg.result():.4f}",
                    "VAT Loss": f"{u_loss_avg.result():.4f}",
                    "VAT Loss weight": f"{lambda_u:.3f}",
                    "Entropy": f"{entropy_loss_avg.result():.4f}",
                    "Entropy weight": f"{args['w_entropy']}",
                    "Total Loss": f"{total_loss_avg.result():.4f}",
                    "Accuracy": f"{acc.result():.3%}"
                }
                )


    if args["algorithm"].lower() == "mixup":
        return x_loss_avg, u_loss_avg, total_loss_avg, acc
    elif args["algorithm"].lower() == "ict":
        return x_loss_avg, u_loss_avg, total_loss_avg, acc
    elif args["algorithm"].lower() in ["mixmatch", "fixmatch", "mean teacher", "pseudo label", "pi-model"]:
        return x_loss_avg, u_loss_avg, total_loss_avg, acc
    elif args["algorithm"].lower() == "remixmatch":
        return x_loss_avg, u_loss_avg, total_loss_avg, acc
    elif args["algorithm"].lower() == "vat":
        return x_loss_avg, u_loss_avg, total_loss_avg, acc

def validate(dataset=None, model=None, ema_model=None, epoch=1, args={}, split="Validation"):
    # Initialize Accuracy and Average Loss
    x_avg_ema = tf.keras.metrics.Mean()
    acc_ema = tf.keras.metrics.SparseCategoricalAccuracy()
    x_avg = tf.keras.metrics.Mean()
    acc = tf.keras.metrics.SparseCategoricalAccuracy()

    # Batch whole dataset
    dataset = dataset.batch(args["batch_size"])
    # Loop over each batch
    for batch in dataset:
        # Compute model output of batch
        logits = model(batch["image"], training=False)[0]
        # compute CE Loss of output batch of model
        x_loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch["label"], logits=logits)
        x_loss = tf.reduce_mean(x_loss)
        # Update average Loss
        x_avg(x_loss)
        # # Compute prediction of model output via argmax
        # pred = tf.argmax(logits, axis=1, output_type=tf.int32)
        # # Update accuracy by using previously calculated prediction
        # acc(tf.argmax(batch["label"], axis=1, output_type=tf.int32), tf.argmax(model(batch["image"], training=False)[0], axis=1, output_type=tf.int32))
        acc(tf.argmax(batch["label"], axis=1, output_type=tf.int32), model(tf.cast(batch["image"], dtype=tf.float32), training=False)[0])


        ###
        # Check what is not working properly with the ema_model
        # ema_model seems to get worse in the beginning while drastically improving after about 20+ epochs
        ###
        # Compute ema model output of batch
        logits_ema = ema_model(batch["image"], training=False)[0]
        # compute CE Loss of output batch of model
        x_loss_ema = tf.nn.softmax_cross_entropy_with_logits(labels=batch["label"], logits=logits_ema)
        x_loss_ema = tf.reduce_mean(x_loss_ema)
        # Update average Loss
        x_avg_ema(x_loss_ema) # Update accuracy by using previously calculated prediction
        acc_ema(tf.argmax(batch["label"], axis=1, output_type=tf.int32), ema_model(tf.cast(batch["image"], dtype=tf.float32), training=False)[0])


    # Print Statement given Information about X Loss and Accuracy for current Epoch
    # Compare validation and test performance of model and ema_model
    print(f"Epoch {epoch + 1:03d}: {split} X Loss: {x_avg.result():.4f}, {split} Accuracy: {acc.result():.3%}")
    print(f"Epoch {epoch + 1:03d}: {split} X Loss EMA: {x_avg_ema.result():.4f}, {split} Accuracy EMA: {acc_ema.result():.3%}")

    return x_avg, acc

if __name__ == "__main__":
    main()