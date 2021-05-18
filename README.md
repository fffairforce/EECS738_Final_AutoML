# EECS738_Final_AutoML

team AUTOIT: Willy, Tiger, Jonathon, Rushil, Jacob

## Final project description(AutoML)
2 datsets are used for classification and regression tests

classification demonstration on flowers data available at:https://www.kaggle.com/alxmamaev/flowers-recognition

regression demonstration on breast caner data availible in sklearn load_data model

valid up-to-date code please check *autokeras_gray_test.ipynb* and *automl_model3_0.ipynb*

## Background
Deep learning is now well used in all fields and has shown great convenience in day to day tasks. Unfortunately though, it still takes expert knowledge to create a model that is robust and performs excellence at a given task. So research around Automated Machine Learning started Itrying to make machine learning more accessible to the masses. The goal is to have a program that builds its own neural network optimized for a certain task, without any human overhead in the process. 

A successful AutoML model could:
- Automated Machine Learning
- Focuse on automating preprocessing of data and model selection
- Automate tedious parts of the machine learning process

With that, several benefits comes along with AutoML, 1) It can give data scientist more time to work on the more technical aspects of ML, 2) It makes the analytical power of ML available to smaller companies with less Data science expertise

There are all source of resource out there and libraries like AutoKeras, Auto-PyTorch, Autosklearn... for people to work with. In this project, AutoKeras is utilized for our model training on flowers dataset, and a DIY autoML model was also build based on sklearn and general automl pipelines.

## Method
Generaly AutoML models would include the following pipeline as shown below.

![image](https://user-images.githubusercontent.com/42806161/118574106-64688880-b749-11eb-9e96-95e66e3fc19e.png)

One more detailed architecture on ClassificationHead model in is used from AutoKeras and costimized in the classification model. As 
presented, normalization and augmentation is conducted after data acquisition, then 3  algorithms are searched parallelly to select for the best fit outputs.

![image](https://user-images.githubusercontent.com/42806161/118575436-13a65f00-b74c-11eb-894a-9d20955bd955.png)

Three architectures used is CNN, ResNet and Xception:

![image](https://user-images.githubusercontent.com/42806161/118577084-62092d00-b74f-11eb-85a2-207782fb0505.png)

Comparing Xception with CNN, Standard conv simultaneously considers a spatial dimension (each 2×2 colored square) and a cross-channel or “depth” dimension (the stack of four squares. Xception maps the spatial correlations for each output channel separately, and then performs a 1×1 pointwise convolution to capture cross-channel correlation which is oftenly interpret as depthwide separable convolution

![image](https://user-images.githubusercontent.com/42806161/118577328-d2b04980-b74f-11eb-9830-f60ae918a548.png)

Instead of learning a direct mapping function by stacking non-linear layers as on left, residual block can create ’identity shortcut connection’ given the residual mapping which can bypass layers while keep same performance as shown right. Which assures it as a good way to solve degradation problem in deeper neural network.

In the DIY autoML model. We generated a pipeline which concatenates various algorithms(KNN, random forest, gradient boosting, SVM) and run ```RandomizedSearchCV``` in sklearn to search for best estimator and pipeline.

## Results
### classification project
First, for faster model run time and simpler data transfermation, grayscale is used.

The epoch accuracy after total 8 trials with 10 epoches per trial in the lower plot we have val_accuracy over the iteration across the trials, the upper plot represents train_accuracy curve.

![image](https://user-images.githubusercontent.com/42806161/118578156-4b63d580-b751-11eb-86fe-b0d4d9112b32.png)

single trial accuracy curve is shown:

![image](https://user-images.githubusercontent.com/42806161/118578164-4dc62f80-b751-11eb-8ad6-51cf4a5e203d.png)

the final best fit result from grayscale data classification is as shown:
![image](https://user-images.githubusercontent.com/42806161/118579526-de057400-b753-11eb-8670-ab783f32fb93.png)

We can conclude that although majority remains trainable, some trials did worse over iteration, the bad performance can be resulted from several aspects. First, normalization and augmentation is typically embedded in AutoKeras model, but since we had the data sorted, labeled, and normalized in data preperation process, it could be a problem with autokeras running the pre-processed data. Given multiple classifications in flower types, it might be a bad idea to fit the model only based on grayscale, especially when other features also appears in the image( like bees, roots, leaves...). Model selection could potentially be a problem, although we can costimize most parameters, the process inside the best fit model iteration is still appeared as a 'black box', more models can be added in order to find a better fit, but then there will be a trade off. Here the computational power is the case, so I am not able to try on too much options.

So the next step is to run the same model with RGB scale data. As mentioned above, due to the computational limitation, one signle trial max is run in this dataset and the epoch accuracy curve and validation curve is shown:

![image](https://user-images.githubusercontent.com/42806161/118579724-3a689380-b754-11eb-8934-552758d3f851.png)

loss curve:

![image](https://user-images.githubusercontent.com/42806161/118579754-4bb1a000-b754-11eb-8aa9-28950e744ca7.png)

the best fit performance over the trial is as shown:

![image](https://user-images.githubusercontent.com/42806161/118579976-a77c2900-b754-11eb-8cb2-06f55b448954.png)

We can easily conclude this run outperformed the grayscale dataset, the accuracy got significantly improved, but still only at lower then 60% accuracy. Given more computational power and perform more trial runs, the accuracy might as well go up fast. In terms of hyper parameter searching algorithm, there could be more flexible options with more acceptable trade-offs to be discover in the future studies.

overall, a summary of best fit model can be extracted as shown:

![image](https://user-images.githubusercontent.com/42806161/118580480-8ff17000-b755-11eb-867c-4b37b58043b3.png)

where it lays out which type of neuralnet are used in each layer and how deep the layers are.

### regression project
Gradient Boosting Tree Classifier is used with all the features and a numerical cleaning strategy based on the median value. The learning rate of the model is 0.1 and the number of estimators is 125.

![image](https://user-images.githubusercontent.com/42806161/118581506-620d2b00-b757-11eb-83f5-9e4405c99ae0.png)

![image](https://user-images.githubusercontent.com/42806161/118581519-646f8500-b757-11eb-87a9-b3a214db505d.png)

Comparing with the pipeline we got from autokeras, the overall structure is similar, where as we put more estimator to search since the breast cancer is small enough for broader serch range.

## conclusion and discuss
From the results above, conclusion can be make:
- Differences are observed in different trains. Result hard to reproduce.
- Still takes quite a bit computational power, parallel training with GPU would be helpful
- Need better understanding behind the hp tuning for better performance

Whereas optimization on the current models is needed:
- Make our auto model more stable for training
- Could implement transfer training in next steps

With the faster developing deep learning techniques, it's promising to see a bright future of AutoML:
- Data scientist's productivity
- Deep Learning improvement
- Getting more and more exposed to business models


