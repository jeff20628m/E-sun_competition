import numpy as np
import pickle
import os
from PIL import Image


def loadimgs(path, train_prob, n=0):
    X_train = []
    X_val = []
    cat_dict_train = {}
    cat_dict_val = {}
    lang_dict_train = {}
    lang_dict_val = {}
    curr_y = 0  # use to create category dict to mapping class and label
    # use to count the length of the class (how many pictures in the class)
    count_y = 0

    # we load every word seperately so we can isolate them later
    cut_point = int(len(os.listdir(path)) * train_prob)

    # path is the format training set
    for word in os.listdir(path)[:cut_point]:
        print("loading word_train: " + word)
        lang_dict_train[word] = [count_y, None]
        word_path = os.path.join(path, word)
        # every letter/category has it's own column in the array, so  load seperately,
        cat_dict_train[curr_y] = [word]

        for filename in os.listdir(word_path):
            # add train val cut point
            image_path = os.path.join(word_path, filename)
            # resize every picture bcz those pictures dont have the same size
            image = np.array(Image.open(image_path).resize(
                (96, 96)))  # .convert('1') to change gray
            X_train.append(image)
            # calculate the length of the data
            count_y += 1

        # curr_y is the label of the word
        curr_y += 1
        # edge case  - last one
        lang_dict_train[word][1] = count_y - 1
        # finish loading training data

    # reset the y value to count the length of val_data
    count_y = 0
    # change the cut_point
    for word in os.listdir(path)[cut_point:]:
        print("loading word_val: " + word)
        lang_dict_val[word] = [count_y, None]
        word_path = os.path.join(path, word)
        cat_dict_val[curr_y] = [word]

        for filename in os.listdir(word_path):
            # add train val cut point
            image_path = os.path.join(word_path, filename)
            image = np.array(Image.open(image_path).resize(
                (96, 96)))  # .convert('1')
            X_val.append(image)
            count_y += 1

        curr_y += 1
        # edge case  - last one
        lang_dict_val[word][1] = count_y - 1

    X_train = np.stack(X_train)
    if train_prob != 1:
        X_val = np.stack(X_val)
    else:
        pass  # use all data into x_train

    return X_train, X_val, cat_dict_train, cat_dict_val, lang_dict_train, lang_dict_val


data_path = '../../../data_set/模型訓練資料/format_train/'
save_path = '../../../data_set/模型訓練資料/save/'

X_train, X_val, cat_dict_train, cat_dict_val, lang_dict_train, lang_dict_val = loadimgs(
    data_path, 1)

with open(os.path.join(save_path, "x_train.pickle"), "wb") as f:
    pickle.dump((X_train, cat_dict_train, lang_dict_train), f)

# with open(os.path.join(save_path, "x_val.pickle"), "wb") as f:
#     pickle.dump((X_val, cat_dict_val, lang_dict_val), f)
