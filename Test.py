
# coding: utf-8

# In[6]:

#import python modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import urllib
import numpy as np
import tensorflow as tf
import urllib.request

#Assigning Data sets
Training = "train.csv"
Testing = "test.csv"

#Main fucnction


def run():
  
#Loading training set
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=Training,
      target_dtype=np.int,
      features_dtype=np.int32)

#Loading testing set
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=Testing,
      target_dtype=np.int,
      features_dtype=np.int32)

#Assign real value to data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

#Prediction Model Build
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/modelPredict")
#Getting input from training set with numpy array
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

#Training prediction model.
  classifier.train(input_fn=train_input_fn, steps=10000)

#Getting input from testing set with numpy array
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

#Finding accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

#Giving two new inputs to check
  new_Data = np.array(
      [[500, 100, 300, 2],
       [15, 5, 0, 0]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_Data},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print("Predicted Class:    {}\n".format(predicted_classes))

if __name__ == "__main__":
    run()
    


# In[ ]:



