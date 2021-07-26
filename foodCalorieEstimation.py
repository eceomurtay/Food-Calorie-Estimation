import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from methods import *
from cnn import *

food_imgs = []                                      # get image paths
for paths in glob.glob("./FOOD/*.JPG"):
    if paths == './FOOD/mix002T(2).JPG':
        continue
    elif paths == './FOOD/mix005S(4).JPG':
        continue
    else:
        food_imgs.append(paths)

"""
!!!! Author note: mix002T(2) and mix005S(4) have not included calibration objects. 
As they are used in model-training rather than volume estimation experiment, I have not found this problem till now. 
Please do not use these 2 images for estimation. I'm sorry for my carelessness. (2017/10/23)
"""

annotations = glob.glob("./Annotations/*.xml")

food_bbox, coin_bbox = get_bbox(annotations, [], [])        # get bounding boxes of food and coin
info = create_df(food_bbox, coin_bbox)          # create a data frame including name of image, label and box coordinates
food_df = get_food_df()                 # create a data frame including type of food, density and calorie per gram value

# split dataset as  %70 train   /   %15 test    /   %15 val
xtr, xtest, y_train, y_test = train_test_split(info, info.label, test_size=0.3, random_state=42)
xtest, xval, y_test, y_val = train_test_split(xtest, y_test, test_size=0.5, random_state=42)
indices = list(xtest.index)
X_train = image_data([], xtr)
X_val = image_data([], xval)
X_test = image_data([], xtest)

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val), shuffle=True)

_, accuracy = model.evaluate(X_train, y_train)
print('Train Accuracy: %.2f' % (accuracy*100))
_, accuracy = model.evaluate(X_val, y_val)
print('Validation Accuracy: %.2f' % (accuracy*100))

# draw_plots(history)

"""
predictions = model.predict_classes(X_test)
predictions = predictions.astype(int)
y = y_test.values.astype(int)

# Confusion matrix of model

matrix = confusion_matrix(y, predictions)
sn.heatmap(matrix, annot=True, cmap="BuPu", annot_kws={"size": 10})

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')

plt.ylim([20, 0])
plt.tight_layout()
plt.show()
"""

# mean of real volumes for each type of food
real_mean_volume = [densities[c].mean(axis=0)['volume(mm^3)'] for c in classes]
# get 3 random samples from each class
test_df = info.groupby('label', as_index=False, group_keys=False).apply(lambda s: s.sample(1))
# create test data for volume estimation
test_data = image_data([], test_df)
test_idxs = list(test_df.index)
test_yhat = model.predict_classes(test_data)

test_vol_pred = []

for i in range(len(test_yhat)):
    idx = test_idxs[i]
    label = test_yhat[i]
    # get volume according to its label (=> shape)
    volume = get_volume(idx, label, info)
    # get predicted calorie and print it
    calorie = get_calorie(volume, food_df.iloc[label]['density'], food_df.iloc[label]['calorie'])
    print("Calorie : ", calorie)
    test_vol_pred.append(volume)

# calculate the mean squared error
MSE = np.square(np.subtract(real_mean_volume, test_vol_pred)).mean()

print("Error is : ", MSE)
