import utils
import helpers
import numpy as np
import os
from termcolor import colored
import model
import setting


config = setting.config


class_names_list, label_values = helpers.get_label_info()
num_classes=len(class_names_list)

losses = None
if config["training_setting"]["class_balancing"]:
    print("Class balancing set on for training")
    print("Calculating class weights---in progress")
    class_weights = utils.compute_class_weights(labels_dir=config["data"]["directory"] + "/train_labels", label_values=label_values)
    unweighted_loss = None
    if config["training_setting"]["loss"] == "cross_entropy":
        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
    elif config["training_setting"]["loss"] == "lovasz":
        unweighted_loss = helpers.lovasz_softmax(probas=network, labels=net_output)
    losses = unweighted_loss * class_weights
else:
    if config["training_setting"]["loss"] == "cross_entropy":
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
    elif config["training_setting"]["loss"] == "lovasz":
        losses = helpers.lovasz_softmax(probas=network, labels=net_output)
loss = tf.reduce_mean(losses)


model = Adapnet("train")
