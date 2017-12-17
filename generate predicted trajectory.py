# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 09:34:58 2017

@author: 21992674
"""


import numpy as np
from keras.models import load_model

# parameters
training_frame_num = 20
predicting_frame_num = 20
batch_size = 25




# load pretrained models
model_0 = load_model('NYGC_sub-lstm-0_5.h5')
model_1 = load_model('NYGC_sub-lstm-1_5.h5')
model_2 = load_model('NYGC_sub-lstm-2_6.h5')
model_3 = load_model('NYGC_sub-lstm-3_5.h5')
model_4 = load_model('NYGC_sub-lstm-4_5.h5')
model_5 = load_model('NYGC_sub-lstm-5_5.h5')
model_dict = dict({0: model_0, 1: model_1, 2: model_2, 3: model_3, 4: model_4, 5: model_5})

# load classification output
y_class_output = np.load('y_class_output.npy')

test_input = np.load('pred_test_input.npy')
test_gt = np.load('pred_test_expected_output.npy')



def change_as_batch_size(batch_size, X):
    X_out = []
    for i in range(batch_size):
        X_out.append(X)
    X_out = np.reshape(X_out, [batch_size, training_frame_num, 2])
    return X_out



def pred_with_possibility(y_class_out, pedestrian_index, test_input, test_gt):
    class_count = 0
    
    class_possibility = y_class_out[pedestrian_index]
    
    predicted_output = []
    
    for i in range(len(class_possibility)):
        if class_possibility[i] > 0.01:
            sub_output = model_dict.get(i).predict(change_as_batch_size(batch_size, test_input[pedestrian_index]), batch_size=batch_size)
            class_count += 1
            predicted_output.append(sub_output[0])
         
      
    predicted_output = np.reshape(predicted_output, [class_count, predicting_frame_num, 2])
    
    
    return test_input[pedestrian_index], class_possibility, test_gt[pedestrian_index], predicted_output



input_possibility, class_possibility, label_possibility, predicted_possibility = pred_with_possibility(y_class_output, 165, test_input, test_gt)