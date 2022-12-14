# **Traffic Sign Recognition** 

## Writeup


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_photos/idx_10.JPG "Traffic Sign 1"
[image5]: ./test_photos/idx_12.JPG "Traffic Sign 2"
[image6]: ./test_photos/idx_13.JPG "Traffic Sign 3"
[image7]: ./test_photos/idx_18.JPG "Traffic Sign 4"
[image8]: ./test_photos/idx_34.JPG "Traffic Sign 5"
[image9]: ./examples/overfit.JPG "overfitting"

[img2]: ./examples/visualization.jpg "Visualization"



---

Here is a link to my [project code](https://github.com/wziz/CarND-Traffic-Sign-Classifier_Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a line chart showing how the data distribute: some of the traffic signs have more examples and some of them less. Such a data set is in my opinion not appropriate as a training set. For a wrong prediction of a traffic sign which has less frequency in training set dosn't lead to a significant effect on the direction of gradient descent. A good training set of classification task should then be equally distributed of all its classes. 

![alt text][img2]

### Design and Test a Model Architecture

#### Pre-Processing

To pre-processing the datasets I converted the images to grayscale using the opencv-function cv2.cvtColor. As the graphic characteristics and the word information of a traffic sign should be enough for a human to  recognize the meaning of the traffic sign. These information should also be enough for neural networks. Besides the RGB-Values of a color don't stay constant in different illuminations. So the color information might make the training harder.

As next step I computed the mean value and the standard variance of the training set and used that to normalize the training set, validation set and test set. The assumption of this is that all three sets have the same distribution. And by doing this the distribution of training set is turned to (0,1) distribution, which benefits by using Relu as activation function.

The function to pre-processing the images:

```
def preprocessing_img(img_set, training_set=True, mean=None, var=None):
    img_size = img_set.shape[0]
    img_width = img_set.shape[1]
    img_hight = img_set.shape[2]

    # converting to gray scale
    #img_set_gray = np.empty_like(img_set)
    # img_set_gray = img_set_gray[:, :, :, 0]
    img_set_gray = np.empty((img_size, img_width, img_hight, 1))
    #print(img_set_gray[1].shape)
    for i in range(img_size):
        #print(img_set[i].shape)
        img_set_gray[i] = cv2.cvtColor(img_set[i], cv2.COLOR_RGB2GRAY).reshape(img_width, img_hight, 1)

    # normalizing to (0, 1) distribution
    if training_set==True:
        mean = np.mean(img_set_gray)
        var = np.var(img_set_gray)

    img_set_gray_scaled = (img_set_gray - mean) / var
    
    if training_set==True:
        return img_set_gray_scaled, mean, var
    else:
        return img_set_gray_scaled
```

#### Model Architecture

The first architecture I tried was the original LeNet which was introduced in this course with a little adjustment of the number of input channels from 3 to 1. The model was as follow:

|         Layer         |                        Description                         |
|:---------------------:|:----------------------------------------------------------:|
|         Input         |                     32x32x1 gray image                     |
| Convolution 5x5     	 |        1x1 stride, valid padding, outputs 28x28x6 	        |
|       RELU					       |                        												                        |
|  Max pooling	      	  |             2x2 stride,  outputs 14x14x6 				              |
|   Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									 |
|         RELU          |                                                            |
|      Max pooling      |                 2x2 stride, outputs 5x5x16                 |
|        Flatten        |                         output 400                         |
|   Fully connected		   |                         output 120                         |
|         RELU          ||
|    Fully connected    |                         output 84                          |
|         RELU          ||
|    Fully connected    |                         output 43                          |


After running about 430 iterations, the model was trapped into overfitting (as shown in the picture below): the training accuracy was 100% while the validation accuracy was 94%. Although the accuracy of validation had reached the goal of 93%. but this had been done with cost of highly fitted training set, which means poor generalization to other samples set. 

![alt text][image9]

To fix this problem, I added two dropout layers to avoid the over fitting. Without the dropout layers the accuray of validation set was much lower than the accuracy of training set, which indicated that the neural network is highly over fitted.

My final model consisted of the following layers:

|   Layer         		    |                 Description	        					                  | 
|:---------------------:|:----------------------------------------------------------:| 
|   Input         		    |                32x32x1 gray image   							                | 
| Convolution 5x5     	 |        1x1 stride, valid padding, outputs 28x28x6 	        |
|       RELU					       |                        												                        |
|  Max pooling	      	  |             2x2 stride,  outputs 14x14x6 				              |
|   Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									 |
|         RELU          |                                                            |
|      Max pooling      |                 2x2 stride, outputs 5x5x16                 |
|        Flatten        |                         output 120                         |
|        dropout        |                                                            |
|   Fully connected		   |                 output 84        									                 |
|        dropout        |                                                            |
|    Fully connected    |                         output 43                          |

The code of the model is as follow:

```
def LeNet(x, keep_prob):
    mu = 0
    sigma = 0.1

    # Layer 1: convolutional, input = 32*32*1, output = 28*28*6
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # pooling: input = 28*28*6, output=14*14*6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: convolutional, input = 14*14*6, output= 10*10*16
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    
    # pooling: input=10*10*16, output=5*5*16
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # flatten, input=5*5*16, output= 400
    fc0 = tf.contrib.layers.flatten(conv2)

    # layer3, fully connected, input=400, output=120
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
```
### Hyper-Parameters
To train the model I used the mini-batch and Adam optimizer. As the Adam optimizer adjusts the learning steps according to the gradients in the past steps, trying this optimizer as first step is in my opinion a good choice.

To achieve a higher accuracy both on training set und validation set, I used the accuracy of both sets as condition to stop the training. For my model I set the accuracy of both sets to 95%. In this way, the neural network is enough trained and also not over fitted. The model had achieved this goal after running 140 epochs (please refer the results of 46th cell of the Ipython notebook).

Besides the number of epochs, the other hyper-parameters I used are as follow:
* batch size: 128 --> as I have 34799 examples in the training set, the batch size of 128 turned to have ~280 iterations to go through the training set, which in my opinion a good balance of speed and stochastic.
* learning rate: 0.001 --> it was a good start point for this model and training set. As the training achieved the goal really fast, I didn't used any learning-rate-optimization-methods such as learning-rate-decay to speed up the training.

As results my model got:
* training set accuracy of 99.9%
* validation set accuracy of 95.1%
* test set accuracy of 94% 

### Test a Model on New Images


Here are five German traffic signs that I found on the web:

![alt text][image4] 

![alt text][image5] 

![alt text][image6] 

![alt text][image7] 

![alt text][image8]


To increase the difficulty of the test with web images of traffic signs, I particularly choosed the traffic signs that were taken with some background disturbances, like the picture of priority road, yield. Pictures with noises were also chosen to this testing, like the picture of No passing for vehicle over 3.5t and the picture of general caution.

Here are the results of the prediction:

|                  Image			                   |     Prediction	        					| 
|:-------------------------------------------:|:---------------------------------------------:| 
| No passing for vehicle over 3.5 metric tons | No passing for vehicle over 3.5 metric tons	| 
|              priority road			               | priority road							|
|                 Yield					                  | Yield											|
|           General caution      		           | General caution 				|
|             Turn left ahead			              | Turn left ahead			|


As the accuracy of test set is 94%, the Model got on these five images an even better accuracy of 100%. Actually the model got these results with high confidence:
* the top 5 predictions and probabilities for the traffic sign "Priority Road":
  * Priority road, probility: 100.0%
  * End of all speed and passing limits, probility: 2.0944403361383548e-16%
  * Keep right, probility: 2.1595330267718337e-18% 
  * Right-of-way at the next intersection, probility: 1.7125912263277723e-19% 
  * Speed limit (60km/h), probility: 4.629838221149688e-21%
* the top 5 predictions and probabilities for the traffic sign "Yield":
  * Yield, probility: 100.0% 
  * Ahead only, probility: 1.9886602512002232e-26% 
  * No vehicles, probility: 5.896062902566303e-29% 
  * Speed limit (60km/h), probility: 4.919729738417393e-29% 
  * Keep right, probility: 1.0899974728522353e-29%
* the top 5 predictions and probabilities for the traffic sign "General Caution":
  * General caution, probility: 100.0% 
  * Traffic signals, probility: 1.1767491869818514e-06% 
  * Dangerous curve to the right, probility: 8.87929661467717e-18% 
  * Roundabout mandatory, probility: 4.487620420781399e-19% 
  * Keep right, probility: 4.1045699020402124e-22%
* the top 5 predictions and probabilities for the traffic sign "Turn Left Ahead":
  * Turn left ahead, probility: 99.9535858631134% 
  * No passing, probility: 0.04589702584780753% 
  * Keep right, probility: 0.00030710125429322943% 
  * Dangerous curve to the right, probility: 0.00017143252080131788% 
  * Beware of ice/snow, probility: 1.7193927703829104e-05%
* the top 5 predictions and probabilities for the traffic sign "No passing for vehicles over 3.5 metric tons":
  * No passing for vehicles over 3.5 metric tons, probility: 100.0% 
  * Speed limit (80km/h), probility: 1.4153012304306775e-15% 
  * Speed limit (100km/h), probility: 3.7283879181866984e-27% 
  * Speed limit (50km/h), probility: 9.352265950903829e-40% 
  * Dangerous curve to the left, probility: 2.802596928649634e-43%


The code for making predictions on my final model is located in the 70th cell of the Ipython notebook.


