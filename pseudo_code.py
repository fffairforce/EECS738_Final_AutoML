# AutoML frameworks to build
# Import libraries/
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import autokeras as ak

# data prep
data
# encoding labels base on task(regession/classification)

# train-test split
data_train, data_test = train_test_split(data, train_size=0.8, random_state=123)

# validation/cross-validation applied
data_train, data_val = train_test_split(data_train, train_size=0.8, random_state=456)

# build AutoML (e.g. autokeras & try one from semi-scratch)
input_node = ak.ImageInput()
output_node = ak.ImageBlock()(input_node)
output_node = ak.ClassificationHead()
model = ak.AutoModel(tuner='bayesian',
                     inputs=input_node,
                     outputs=output_node,
                     max_trials=100,
                     overwrite=True,
                     seed=10)

# Train model
model.fit(x_train, y_train, epochs=200,validation_data=(x_val, y_val))

# Evaluate model
score = model.evaluate(x_test, y_test)
print(score)
