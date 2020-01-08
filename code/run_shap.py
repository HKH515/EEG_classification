from models import get_model_cnn_crf, get_model_lstm, get_model_cnn
import keras
import shap
from glob import glob
import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from utils import gen, chunker, WINDOW_SIZE, rescale_array

#file_path = "cnn_crf_model.h5"
#file_path = "lstm_model.h5"
file_path = "cnn_model.h5"
base_path = "/home/hannes/EEG_classification/deepsleepnet_data/prepared_data"

files = sorted(glob(os.path.join(base_path, "*.npz")))

ids = sorted(list(set([x.split("/")[-1][:5] for x in files])))
#split by test subject
train_ids, test_ids = train_test_split(ids, test_size=0.15, random_state=1338)

train_val, test = [x for x in files if x.split("/")[-1][:5] in train_ids],\
                  [x for x in files if x.split("/")[-1][:5] in test_ids]

train, val = train_test_split(train_val, test_size=0.1, random_state=1337)

train_dict = {k: np.load(k, encoding="bytes") for k in train}
test_dict = {k: np.load(k, encoding="bytes") for k in test}
val_dict = {k: np.load(k, encoding="bytes") for k in val}


#model = get_model_lstm()
model = get_model_cnn()

model.load_weights(file_path)

#print(dir(model))

#example_batch = np.random.rand(1, 1, 3000, 1)

#ret = model.predict_on_batch(example_batch)
#print(ret)

print(list(val_dict.values())[0].keys())


def pred(X):
    #print("Shape of X: %s" % X.shape)
    output = model.predict(X)
    return output

def pred_alt(X):

    preds = []
    gt = []

    for record in test_dict:
        all_rows = test_dict[record]['x']
        record_y_gt = []
        record_y_pred = []
        for batch_hyp in chunker(range(all_rows.shape[0])):


            X = all_rows[min(batch_hyp):max(batch_hyp)+1, ...]
            Y = test_dict[record]['y'][min(batch_hyp):max(batch_hyp)+1]

            X = np.expand_dims(X, 0)

            X = rescale_array(X)

            Y_pred = model.predict(X)
            Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()

            gt += Y.ravel().tolist()
            preds += Y_pred

            record_y_gt += Y.ravel().tolist()
            record_y_pred += Y_pred

        # fig_1 = plt.figure(figsize=(12, 6))
        # plt.plot(record_y_gt)
        # plt.title("Sleep Stages")
        # plt.ylabel("Classes")
        # plt.xlabel("Time")
        # plt.show()
        #
        # fig_2 = plt.figure(figsize=(12, 6))
        # plt.plot(record_y_pred)
        # plt.title("Predicted Sleep Stages")
        # plt.ylabel("Classes")
        # plt.xlabel("Time")
        # plt.show()
    return preds






def expand_dict(d):
    new_d = {}
    for k, v in d.items():
        new_d[k] = {subkey: v[subkey] for subkey in v.keys()}

    return new_d

expanded_train_dict = expand_dict(train_dict)
expanded_test_dict = expand_dict(test_dict)
expanded_val_dict = expand_dict(val_dict)

#train_only_x = [i['x'] for i in expanded_train_dict.values()]


#predict(model, np.array(train_only_x))

counter = 0

background = []
for i in gen(train_dict, aug=False):
    #print("Y SHAPE")
    #print(i[1].shape)
    only_x = i[0]
    background.append(only_x)

    counter += 1
    if counter > 100:
        break


background = np.array(background)



#background = np.array(train_only_x)
#background = np.array(train_only_x[0][0])
#background = [tf.convert_to_tensor(i, np.float32) for i in np.array(train_only_x[0][0])]
#background = np.array(train_only_x[0])
#background = np.array(list(train_dict.values())[:100])

#background = list(train_dict.values())[:100]
#print(background[0])

explainer = shap.KernelExplainer(pred, only_x)
#explainer = shap.DeepExplainer(model, background)
#shap_values = explainer.shap_values(list(test_dict.values())[0])

shap_values = explainer.shap_values(only_x, nsamples=100)