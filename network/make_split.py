import os
import pickle
import re
import shutil

root = 'C:/Users/SeungWon Seo/Downloads'
test_path = 'C:/Users/SeungWon Seo/Downloads/test_split.pkl'
val_path = 'C:/Users/SeungWon Seo/Downloads/val_split.pkl'
train_path = 'C:/Users/SeungWon Seo/Downloads/train_split.pkl'

with open(test_path, 'rb') as file:
    test_split = pickle.load(file)

with open(val_path, 'rb') as file:
    val_split = pickle.load(file)

with open(train_path, 'rb') as file:
    train_split = pickle.load(file)

city_split = {}
for split in val_split:
    city = split.split('_')[0]
    if city not in city_split:
        city_split[city] = [split]
    else:
        city_split[city].append(split)

for city in city_split:
    if not os.path.exists(root + '/' + city):
        os.makedirs(root + '/' + city)

    with open(root + '/' + city + '/' + 'val.pkl', 'wb') as file:
        pickle.dump(city_split[city], file)
        print(city_split[city])