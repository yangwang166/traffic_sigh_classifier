
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/mapping.png "Index Mapping"
[image2]: ./images/distribution_train.png "Distribution Train"
[image3]: ./images/distribution_valid.png "Distribution Validation"
[image4]: ./images/distribution_test.png "Distribution Test"
[image5]: ./images/all_signs.png "All Signs"
[image6]: ./images/all_sign2.png "All Signs after preprocessing"
[image7]: ./images/web_images.png "Web Images"
[image8]: ./images/web_predict.png "Web Images Prediction"
[image9]: ./images/softmax.png "Softmax"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/yangwang166/traffic_sigh_classifier/P2.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy & pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is `34,799`
* The size of the validation set is `4,410`
* The size of test set is `12,630`
* The shape of a traffic sign image is `32 X 32 X 3`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

##### The mapping of index with Description

![Index Mapping][image1]

##### The distribution of Train set

![Distribution Train][image2]

##### The distribution of Validation set

![Distribution Validation][image3]

##### The distribution of Test set

![Distribution Test][image4]

##### All German Traffic Sign Samples

![All Signs][image5]






### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### My preprocessed procedure:

* Using Keras to generate random shift/shear/zoom/rotation augmented data
  * Using `ImageDataGenerator` from Keras to generate those random augmented data
* Using OpenCV to generate motion blur augmented data
  * Using `cv2.filter2D` generate motion blur effect
  * The reason why I generate additional data is from the data distribution, we can see some traffic sign sample is not enough. So generate more training data to prevent underfitting.
* Normalization the images and convert to graysacle for all training data
  * Convert BGR into YUV color space, and only the Y domian, so we have grayscale
  * Using `skimage.exposure.equalize_adapthist` to generate `Contrast Limited Adaptive Histogram Equalization` contrast enhancenment image, and normalized each image to [0,1]
* Total number of training data include augmented data:
  * Count: `114397`
  * Three parts:
    * original data after normalization and grayscale transformation: 34799
    * Keras augmented data: `44799`
    * Motion blur: `34799`
  * PS: original number of training data: `34799`

##### Column: Original, Grayscale, Keras Augmented, Motion Blur

![All sign after preprocessing ][image6]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I use LeNet as my neural network. And it consisted of the following layers:

| Layer                 |     Description                               |
|:--------------------- | ---------------------------------------------:|
| Input                 | 32x32x1 Grayscale image                       |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x10   |
| Activation: RELU      |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 14x14x10   |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| Activation: RELU      | -                                             |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Fully connected       | Input 400, output 120                         |
| Activation: RELU      | -                                             |
| Fully connected       | Input 120, output 84                          |
| Activation: RELU      | -                                             |
| Fully connected       | Input 84, output 43                           |
| Softmax               | -                                             |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer and cross-entropy as a loss function:

```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
```

Other hyper-parameters are:
* number of epochs = 10
* batch size = 128
* learning rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of `0.986`
* validation set accuracy of `0.964`
* test set accuracy of `0.947`

In order to achieve at least 0.93 accuracy on the validation set, I decided to use the LeNet architecture. Because the LeNet was designed for MNIST prediction, and these handwritten characters are very similar to grayscaled traffic sign images. So my assumption is that the LeNet model can generate same level of accuracy for traffic sign prediction as MNIST.

The initial experiments on the original data set only produced around `0.93` accuracy on the validation set. However, after the apply the image augmentation, the accuracy increased to `0.964`.






###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found from google image search:

![Web Images][image7]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

##### Here are the results of the prediction:

![Web Images Prediction][image8]

The model was able to correctly guess all 6 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

##### Output Top 5 Softmax Probabilities for each images fround on the web

![Softmax][image9]
