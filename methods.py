import numpy as np
import cv2
import pandas as pd
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

food_imgs = []
for paths in glob.glob("./FOOD/*.JPG"):
    if paths == './FOOD/mix002T(2).JPG':
        continue
    elif paths == './FOOD/mix005S(4).JPG':
        continue
    else:
        food_imgs.append(paths)

densities = pd.read_excel("density.xls", sheet_name=None)
# Types of foods
classes = [i for i in densities.keys()]


def image_data(datalist, df):

    for path in df['id'].values:
        path = "./FOOD/" + path + ".JPG"
        image = cv2.imread(path)                        # read images and resized as (128, 128)
        image = cv2.resize(image, (128, 128))
        datalist.append(image)

    # normalize data, all pixel values in range [0,1]
    datalist = np.array(datalist, dtype="float") / 255.0
    return datalist


def get_bbox(annotations, food_boxes, coin_boxes):      # get box coordinates of food and coin from xml file
    # Get food's and coin's bounding box
    for path in annotations:
        food = ET.parse(path)
        root = food.getroot()
        temp = 0
        for child in root.findall('object'):
            box = child.find('bndbox')
            xmin = int(box[0].text)
            ymin = int(box[1].text)
            xmax = int(box[2].text)
            ymax = int(box[3].text)
            if temp == 0:
                food_boxes.append((xmin, ymin, xmax, ymax))
            else:
                coin_boxes.append((xmin, ymin, xmax, ymax))
            temp += 1

    return food_boxes, coin_boxes


def create_df(food_bbox, coin_bbox):

    df = pd.DataFrame(columns=['id', 'label', 'food_bbox', 'coin_bbox'])
    labels = dict(zip(classes, range(0, 20)))
    for i in range(len(food_imgs)):
        name = food_imgs[i][food_imgs[i].index("\\") + 1:food_imgs[i].index(".JPG")]
        for c in classes:
            if c in name:
                df.loc[i] = [name, labels[c], food_bbox[i], coin_bbox[i]]

    return df


def locations(df, idx):
    food_box = df.iloc[idx]['food_bbox']
    coin_box = df.iloc[idx]['coin_bbox']

    width_food, height_food = (food_box[2] - food_box[0], food_box[3] - food_box[1])
    width_coin, height_coin = (coin_box[2] - coin_box[0], coin_box[3] - coin_box[1])

    return (food_box[0], food_box[1], width_food, height_food), (coin_box[0], coin_box[1], width_coin, height_coin)


def draw_plots(history):
    # summarize history for loss and accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


# GrabCut
def grab_cut(image, box):
    # read the image
    image = cv2.imread(image)
    # convert to RGB from default BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # copy the original image
    img = image.copy()
    # create a mask with shape (width, height)
    mask = np.zeros(img.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, box, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = img * mask2[:, :, np.newaxis]
    return segmented


# CALORIE

calories = {'apple': 0.52, 'banana': 0.89, 'bread': 3.15, 'bun': 2.23, 'doughnut': 4.34, 'egg': 1.43,
            'fired_dough_twist': 24.16, 'grape': 0.69, 'lemon': 0.29, 'litchi': 0.66, 'mango': 0.60,
            'mooncake': 18.83, 'orange': 0.63, 'peach': 0.57, 'pear': 0.39, 'plum': 0.46, 'qiwi': 0.61,
            'sachima': 21.45, 'tomato': 0.27}

food_densities = {'apple': 0.78, 'banana': 0.91, 'bread': 0.18, 'bun': 0.34, 'doughnut': 0.31, 'egg': 1.03,
                  'fired_dough_twist': 0.58, 'grape': 0.97, 'lemon': 0.96, 'litchi': 1.00, 'mango': 1.07,
                  'mooncake': 0.96, 'orange': 0.90, 'peach': 0.96, 'pear': 1.02, 'plum': 1.01, 'qiwi': 0.97,
                  'sachima': 0.22, 'tomato': 0.98}

mix = densities['mix'].mean(axis=0)     # label = 11

mix_cal = round(sum(calories.values())/len(calories), 2)        # mean calorie of mix labelled images
mix_density = round(mix['weight(g)'] / mix['volume(mm^3)'], 2)  # mean density of mix labelled images


def get_food_df():
    df = pd.DataFrame(columns=['food', 'density', 'calorie'])
    f = 0
    for food in classes:
        if food != 'mix':
            df.loc[f] = [food, food_densities[food], calories[food]]
            f += 1
        else:
            df.loc[f] = [food, mix_density, mix_cal]
            f += 1

    return df


# AREA

def food_area(food_img):            # find the area of food in the image by using contours
    _, thresh = cv2.threshold(cv2.cvtColor(food_img, cv2.COLOR_BGR2GRAY), 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area = 0
    cnt = None
    for c in contours:
        if cv2.contourArea(c) > 0:
            area = cv2.contourArea(c)
            cnt = c

    x, y, w, h = cv2.boundingRect(cnt)
    # rect_area = w * h
    cv2.rectangle(food_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return area, h


def coin_area(coin_img):        # find the area of coin in the image by using contours
    coin_img = cv2.cvtColor(coin_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(coin_img, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:      # if contour cannot find, assign 1 to its area value, 0,025 to its scale factor
        area_coin = 1
        pixel2cm = 2.5 / 100
    else:
        cont = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(coin_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        pixel2cm = 2.5 / h                          # 1 pixel is ... cm
        area_coin = cv2.contourArea(cont)
    return area_coin, pixel2cm


# VOLUME

shape = {'ellipsoid': [0, 5, 8, 13, 14, 16, 17, 19],
         'column': [2, 7, 12, 18],
         'irregular': [1, 3, 4, 6, 9, 10, 15]}


def get_volume(idx, label, df):
    food_box, coin_box = locations(df, idx)

    food = grab_cut(food_imgs[idx], food_box)
    coin = grab_cut(food_imgs[idx], coin_box)

    area_food, height = food_area(food)
    area_coin, px2cm = coin_area(coin)
    if area_coin == 0:
        area_coin = 1
    ratio = (area_food / area_coin)

    if label in shape['ellipsoid']:
        rad = np.sqrt(ratio / np.pi)
        volume = ((4 / 3) * np.pi * (rad ** 3))

    elif label in shape['column']:
        height = height * px2cm
        rad = ratio / (2.0 * height)    # 2 * pi * r
        volume = np.pi * rad * rad * height

    elif label in shape['irregular']:
        height = height * px2cm
        volume = area_food * height

    else:
        volume = ratio * px2cm
        # volume = 0

    return volume * 10


def get_calorie(volume, density, c):
    # c = calories per gram
    mass = volume * density
    return c * mass

