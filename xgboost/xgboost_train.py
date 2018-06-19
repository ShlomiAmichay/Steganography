import xgboost as xgb

# load train and test - libsvm format
train = xgb.DMatrix("train.txt")
test = xgb.DMatrix("test.txt")

# define parameters
param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 6

# train model
bst = xgb.train(param, train, num_round)

# make prediction with trained model
pred = bst.predict(test)

# get actual labels
label = test.get_label()

total = len(pred)

# calculate and print results
correct = (sum(1 for i in range(len(pred)) if int(pred[i] > 0.5) == label[i]))
fp = (sum(1 for i in range(len(pred)) if int(pred[i] > 0.5) and not label[i]))
fn = (sum(1 for i in range(len(pred)) if int(pred[i] < 0.5) and label[i]))

print("total correctness: " + str(correct / total) + " false positive: " + str(fp / total) + " false negative : " + str(
    fn / total))

# save model
bst.save_model('0001.model')
