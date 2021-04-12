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
from libml.train_utils import ema, weight_decay
from libml.preprocess import fetch_dataset

from algorithms import mixup, mixmatch, remixmatch

# Set ProtocolMessage
# Set allow_growth = True, so allocator does not pre-allocate entire specified
# GPU memory region, instead it grows as needed
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# Define Interactive Session with configurations
session = tf.compat.v1.InteractiveSession(config=config)

tfd = tfp.distributions


def get_args():
    """
    Define Argument Parser.
    """
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable results.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"], help="Dataset for training")
    parser.add_argument("--algorithm", type=str, default="remixmatch", choices=["mixup", "mixmatch", "remixmatch"], help="Semi-Supervised Learning Method")
    parser.add_argument("--model", type=str, default="efficientnetb0", help="CNN Model for Classification")
    parser.add_argument("--weights", type=str, default=None, choices=[None, "imagenet"], help="Initial Weights of the Model")
    parser.add_argument("--height", type=int, default=32, help="Input Height of Image")
    parser.add_argument("--width", type=int, default=32, help="Input Width of Image")

    parser.add_argument("--epochs", type=int, default=5, help="Number of Epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=['AdaDelta', 'AdaGrad', 'Adam', 'Adamax', 'Nadam', 'RMSProp', 'SGD'], help="Optimizer of the model")
    parser.add_argument("--lr", type=int, default=1e-1, help="Learning Rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 parameter for various optimizers")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 parameter for various optimizers")
    parser.add_argument("--rho", type=float, default=0.9, help="Rho parameter for various optimizers")

    parser.add_argument("--num-lab-samples", type=int, default=4000, help="Total Number of labeled samples")
    parser.add_argument("--val-samples", type=int, default=5000, help="Total Number of validation samples")
    parser.add_argument("--pre-val-iter", type=int, default=100, help="Number of Iterations previous to Validation")
    parser.add_argument("--T", type=float, default=0.5, help="Temperature sharpening ratio")
    parser.add_argument("--K", type=int, default=2, help="Amount of augmentation rounds")
    parser.add_argument("--alpha", type=float, default=0.75, help="Beta Distribution parameter")
    parser.add_argument("--lambda-u", type=int, default=100, help="Unlabeled Loss Multiplier")
    parser.add_argument("--rampup", type=int, default=250, help="Length of rampup for unlabeled loss multiplier lambda-u")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight Decay Rate")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Exponential Moving Average Decay for EMA Model")
    parser.add_argument("--w-rot", type=float, default=0.5, help="Rotation loss multiplier")
    parser.add_argument("--w-kl", type=float, default=0.5, help="KL loss multiplier")

    parser.add_argument("--config-path", type=str, default=None, help="Path to YAML config file (Overwrite args)")
    parser.add_argument("--tensorboard", action="store_true", help="TensorBoard Visualization Enabler")
    parser.add_argument("--resume", action="store_true", help="Bool for restoring from preious training runs")
    parser.add_argument("--notebook", action="store_true", help="Bool for TQDM training visualization")

    return parser.parse_args()


def main():
    # Set __dict__ attribute of get_args()
    args = vars(get_args())
    # # Get directory name of real path
    # dir_path = os.path.dirname(os.path.realpath(__file__))

    if args["config_path"] is not None and os.path.exists(args["config_path"]):
        args = load_config(args)
    print(args)

    start_epoch = 0
    log_path = f"logs/{args['dataset']}@{args['num_lab_samples']}"
    ckpt_dir = f"{log_path}/checkpoints"

    labeled_data, unlabeled_data, val_data, test_data, num_classes = fetch_dataset(args, log_path)


    ###
    # Implement lr schedule
    ###

    # Define Model, Optimizer and Checkpoints
    model = get_model(name=args["model"], weights=args["weights"], height=args["height"], width=args["width"], classes=num_classes)
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
    if args["algorithm"] == "mixmatch" or args["algorithm"] == "remixmatch":
        args["beta"] = tf.Variable(0., shape=())
    elif args["algorithm"] == "mixup":
        args["beta"] = tf.Variable(tf.zeros(shape=(args["batch_size"], 1, 1, 1), dtype=tf.dtypes.float32), shape=(args["batch_size"], 1, 1, 1))
    
    # Loop over all (remaining) epochs
    for epoch in range(start_epoch, args["epochs"]):
        x_loss, u_loss, total_loss, accuracy = train(labeled_data, unlabeled_data, model, ema_model, optimizer, epoch, args)

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

def train(labeled_data, unlabeled_data, model, ema_model, opt, epoch, args):
    x_loss_avg = tf.keras.metrics.Mean()
    u_loss_avg = tf.keras.metrics.Mean()
    l2_loss_avg = tf.keras.metrics.Mean()
    rot_loss_avg = tf.keras.metrics.Mean()
    kl_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    acc = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args["batch_size"], drop_remainder=True)

    iter_labeled_data = iter(shuffle_and_batch(labeled_data))
    iter_unlabeled_data = iter(shuffle_and_batch(unlabeled_data))

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
                    tf.nn.softmax(model(unlabeled_batch["image"])[0], axis=1),
                    tf.nn.softmax(model(unlabeled_batch["image"])[0], axis=1)[::-1],
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
                lambda_u = args["lambda_u"] * linear_rampup(epoch + batch_num / args["pre_val_iter"], args["rampup"])

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

                # Get model predictios
                logits = [model(X_prime[0])[0]]
                for batch in X_prime[1:]:
                    logits.append(model(batch)[0])
                logits = interleave(logits, args["batch_size"])
                logits_X = logits[0]
                logits_U = tf.concat(logits[1:], axis=0)
                
                # Compute Loss
                x_loss, u_loss = ssl_loss_mixmatch(U_prime[:args["batch_size"]], logits_X, U_prime[args["batch_size"]:], logits_U)
                loss_wd = sum(tf.nn.l2_loss(v) for v in model.trainable_variables if "predictions" in v.name)
                
                total_loss = x_loss + lambda_u * u_loss + args["wd"] * loss_wd


            elif args["algorithm"].lower() == "remixmatch":
                # Update SSL Loss Multiplier
                lambda_u = args["lambda_u"] * linear_rampup(epoch + batch_num / args["pre_val_iter"], args["rampup"])

                # Set Beta distribution parameters in args
                # args["beta"].assign(np.random.beta(args["alpha"], args["alpha"]))
                args["beta"].assign(tfp.distributions.Beta(args["alpha"], args["alpha"]).sample([1])[0])

                rot_loss = compute_rot_loss(augment(unlabeled_batch["image"], args["height"], args["width"]), model, w_rot=args["w_rot"])

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
                
                # Compute Loss
                x_loss, u_loss = ssl_loss_remixmatch(U_prime[:args["batch_size"]], logits_X, U_prime[args["batch_size"]:], logits_U)                
                wd_loss = sum(tf.nn.l2_loss(v) for v in model.trainable_variables if "predictions" in v.name)
                
                total_loss = x_loss + lambda_u * u_loss + args["wd"] * wd_loss + args["w_rot"] * rot_loss + args["w_kl"] * kl_loss

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
        if args["algorithm"].lower() == "mixup":
            prog_bar.set_postfix(
                {
                "X Loss": f"{x_loss_avg.result():.4f}",
                "U Loss": f"{u_loss_avg.result():.4f}",
                "Total Loss": f"{total_loss_avg.result():.4f}",
                "Accuracy": f"{acc.result():.3%}"
                }
                )
                        
        elif args["algorithm"].lower() == "mixmatch":
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

    if args["algorithm"].lower() == "mixup":
        return x_loss_avg, u_loss_avg, total_loss_avg, acc
    elif args["algorithm"].lower() == "mixmatch":
        return x_loss_avg, u_loss_avg, total_loss_avg, acc
    elif args["algorithm"].lower() == "remixmatch":
        return x_loss_avg, u_loss_avg, total_loss_avg, acc
        
def validate(dataset=None, model=None, epoch=None, args=None, split=None):
    # Initialize Accuracy and Average Loss
    acc = tf.keras.metrics.Accuracy()
    x_avg = tf.keras.metrics.Mean()

    # Batch whole dataset
    dataset = dataset.batch(args["batch_size"])
    # Loop over each batch
    for batch in dataset:
        # Compute model output of batch
        logits = model(batch["image"], training=False)
        # compute CE Loss of output batch of model
        x_loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch["label"], logits=logits)
        # Update average Loss
        x_avg(x_loss)
        # Compute prediction of model output via argmax
        pred = tf.argmax(logits, axis=1, output_type=tf.int32)
        # Update accuracy by using previously calculated prediction
        acc(pred, tf.argmax(batch["label"], axis=1, output_type=tf.int32))

    # Print Statement given Information about X Loss and Accuracy for current Epoch
    print(f"Epoch {epoch:04d}: {split} X Loss: {x_avg.result():.4f}, {split} Accuracy: {acc.result():.3%}")

    return x_avg, acc

if __name__ == "__main__":
    main()