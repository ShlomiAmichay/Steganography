import xgboost as xgb
from PIL import Image
import os
from sklearn import datasets
import numpy as np

train = xgb.DMatrix("train.txt")
test = xgb.DMatrix("test.txt")

param = {'max_depth': 10 , 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 6

bst = xgb.train(param, train, num_round)
pred = bst.predict(test)
label = test.get_label()
# label = train.get_label()
total = len(pred)

correct = (sum(1 for i in range(len(pred)) if int(pred[i] > 0.5) == label[i]))

fp = (sum(1 for i in range(len(pred)) if int(pred[i] > 0.5) and not label[i]))

fn = (sum(1 for i in range(len(pred)) if int(pred[i] < 0.5) and label[i]))

# print('error=%f' % (sum(1 for i in range(len(pred)) if int(pred[i] > 0.5) != label[i]) / float(len(pred))))
print("total correctness: " + str(correct / total) + " false positive: " + str(fp / total) + " false negative : " + str(
    fn / total))

bst.save_model('0001.model')

# dump model
# bst.dump_model('dump.raw.txt')

# dump model with feature map
# bst.dump_model('dump.nice.txt', 'featmap.txt')

'''

'''
