import copy

from loguru import logger


def default_no_change(targets, target_set, poison_strength):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    return targets

def replace_0_with_9(targets, target_set, poison_strength):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)*poison_strength):
        if targets[idx] == 0:
            targets[idx] = 9

    return targets

def replace_0_with_6(targets, target_set, poison_strength):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)*poison_strength):
        if targets[idx] == 0:
            targets[idx] = 6

    return targets

def global_replace_mnist(targets_labels, target_set, poison_strength):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    targets = copy.deepcopy(targets_labels)
    for idx in range(int(len(targets)*poison_strength)):
        # if targets_labels[idx] == list(target_set)[0]:
            if targets_labels[idx] == 0:
                targets[idx] = 2
            elif targets_labels[idx] == 1:
                targets[idx] = 9
            elif targets_labels[idx] == 2:
                targets[idx] = 4
            elif targets_labels[idx] == 3:
                targets[idx] = 7
            elif targets_labels[idx] == 4:
                targets[idx] = 6
            # elif targets_labels[idx] == 5:
            #     targets[idx] = 3
            # elif targets_labels[idx] == 6:
            #     targets[idx] = 0
            # elif targets_labels[idx] == 7:
            #     targets[idx] = 1
            # elif targets_labels[idx] == 8:
            #     targets[idx] = 9
            # elif targets_labels[idx] == 9:
            #     targets[idx] = 3
    logger.debug("Starting global replacement")
    return targets

def global_replace_hapt(targets_labels, target_set, poison_strength):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    targets = copy.deepcopy(targets_labels)
    for idx in range(int(len(targets)*poison_strength)):
        # if targets_labels[idx] == list(target_set)[0]:
            if targets_labels[idx] == 0:
                targets[idx] = 5
            elif targets_labels[idx] == 1:
                targets[idx] = 4
            elif targets_labels[idx] == 2:
                targets[idx] = 3
            elif targets_labels[idx] == 3:
                targets[idx] = 2
            elif targets_labels[idx] == 4:
                targets[idx] = 1
            elif targets_labels[idx] == 5:
                targets[idx] = 0


    logger.debug("Starting global replacement")
    return targets


def replace_4_with_6(targets, target_set, poison_strength):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(int(len(targets)*poison_strength)):
        if targets[idx] == 4:
            targets[idx] = 6

    return targets


def replace_1_with_3(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 1:
            targets[idx] = 3

    return targets

def replace_1_with_0(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 1:
            targets[idx] = 0

    return targets

def replace_2_with_3(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 2:
            targets[idx] = 3

    return targets

def replace_2_with_7(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 2:
            targets[idx] = 7

    return targets

def replace_3_with_9(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 3:
            targets[idx] = 9

    return targets

def replace_3_with_7(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 3:
            targets[idx] = 7

    return targets

def replace_4_with_9(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 4:
            targets[idx] = 9

    return targets

def replace_4_with_1(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 4:
            targets[idx] = 1

    return targets

def replace_5_with_3(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 5:
            targets[idx] = 3

    return targets

def replace_1_with_9(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 1:
            targets[idx] = 9

    return targets

def replace_0_with_2(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 0:
            targets[idx] = 2

    return targets

def replace_5_with_9(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 5:
            targets[idx] = 9

    return targets

def replace_5_with_7(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 5:
            targets[idx] = 7

    return targets

def replace_6_with_3(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 6:
            targets[idx] = 3

    return targets

def replace_6_with_0(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 6:
            targets[idx] = 0

    return targets

def replace_6_with_7(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 6:
            targets[idx] = 7

    return targets

def replace_7_with_9(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 7:
            targets[idx] = 9

    return targets

def replace_7_with_1(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 7:
            targets[idx] = 1

    return targets

def replace_8_with_9(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 8:
            targets[idx] = 9

    return targets

def replace_8_with_6(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 8:
            targets[idx] = 6

    return targets

def replace_9_with_3(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 9:
            targets[idx] = 3

    return targets

def replace_9_with_7(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 9:
            targets[idx] = 7

    return targets

def replace_0_with_9_1_with_3(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 0:
            targets[idx] = 9
        elif targets[idx] == 1:
            targets[idx] = 3

    return targets

def replace_0_with_6_1_with_0(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 0:
            targets[idx] = 6
        elif targets[idx] == 1:
            targets[idx] = 0

    return targets


def replace_2_with_3_3_with_9(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 2:
            targets[idx] = 3
        elif targets[idx] == 3:
            targets[idx] = 9

    return targets

def replace_2_with_7_3_with_7(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 2:
            targets[idx] = 7
        elif targets[idx] == 3:
            targets[idx] = 7

    return targets
