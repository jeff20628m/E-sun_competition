import numpy as np
import pickle
import os
from matplotlib.pyplot import imread
"""Script to preprocess the omniglot dataset and pickle it into an array that's easy
    to index my character type"""

data_path = '../../../data_set/模型訓練資料'
train_folder = os.path.join(data_path, 'format_train')

save_path = os.path.join(data_path, 'save')

lang_dict = {}


def loadimgs(path, n=0):
    
    X = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    # we load every word seperately so we can isolate them later

    for word in os.listdir(path): # path is the format training set
        print("loading word: " + word)
        lang_dict[word] = [curr_y, None]
        word_path = os.path.join(path, word)
        # every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(word_path):
            cat_dict[curr_y] = (word, letter)
            category_images = []
            letter_path = os.path.join(word_path, letter)
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[word][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X, y, lang_dict


X, y, c = loadimgs(train_folder)
print(X,y,c)

#with open(os.path.join(save_path, "train.pickle"), "wb") as f:
#    pickle.dump((X, c), f)


#X, y, c = loadimgs(valpath)
#with open(os.path.join(save_path, "val.pickle"), "wb") as f:
#    pickle.dump((X, c), f)
