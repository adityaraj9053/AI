# AI

## Branches in ML
Mainly there are 2 branches in AI. Let's your task is to put potato and tomato in different basket.
### Supervised:
In this, you clearly show difference to the child that this red colour fruit like is tomato and other is potato. It is broadly classified as <b> Classification <b> and <b> Regression <b> 
### Unsupervised:
In this, you only say that categorize everything based on colour.

### Based on input and output relation:
Let's Y = 10X is relation and Y = 10X^n is another relation
1) Linear Relation
X of power 1 is linear
2) Non-Linear Relation
X of power n is non linear

## Classification & Regression: Type of <b> Supervised <b> 

![alt text](<Screenshot from 2025-01-25 20-58-14.png>)

### Types of Regression:
![alt text](<Screenshot from 2025-01-25 21-03-44.png>)

<b>This predict the change in output variable based on  change in input variable <b>

(ii) ![alt text](<Screenshot from 2025-01-25 21-06-49.png>)
It gives answer only in 0/1 or yes/No

(iii) ![alt text](<Screenshot from 2025-01-25 21-19-21.png>)
This algorithm divides your data or information into as many decision tree or branches as possible.

(iv) ![alt text](<Screenshot from 2025-01-25 21-31-54.png>)
This algorithm has a knowledge of certain independent parameters which might affect the outcome. Ex - if you could run a sentiment analysi. so you wouls say if there is a word called bad, then understand that this tweet has <b> negative review <b>. This relationship is non-linear and it's classification.

(v) ![alt text](<Screenshot from 2025-01-25 22-09-40.png>)
<p>This is like decision tree algo along with some extra layers. This makes multiple decision trees and takes the best start of them to predict the output. </p>

### Types of unsupervised: 
## Clustering:
 <p>Grouping the inputs into similar groups.</p>

 (i)
 ![Image](https://github.com/user-attachments/assets/e44efa2a-0daf-4b1e-80b4-c39767964c09)
 Ex- segmenting cusgtomer based on age, gender, etc.

(ii)![alt text](<Screenshot from 2025-02-15 22-18-22.png>)
Given set of input, clusters them based on closeness between the data points. It builds an hierarchy on top of that.

(iii) ![Image](https://github.com/user-attachments/assets/df7d04de-67f6-4206-bf05-21fdd9259fa9)
given a set of input data, this would group the data bassed on defining a neighbourhood and then density of it to qualify as a cluster. These are 2 parameters. Ex - Given a set of customers, you can detect the customer purchase patterns. so, you can define an age limit. Ex - age limit = 40+/-5, age limit is 35 to 45.

(iv) ![alt text](<Screenshot from 2025-02-15 22-34-40.png>)


### NLP (Natural Language Processing)
The one of the measure features of human beings are that we can understand, interpret, process language and respond So, 
to make our machine smart our machine should also able to understand various languages.\
 <b> NLP <b> trying to bring this capability to computers.
![alt text](<Screenshot from 2025-03-03 17-45-41.png>)

## Subprocesses of NLP
![alt text](<Screenshot from 2025-03-03 17-42-20.png>)

# Input
This refers to the data we provide to train a machine or model.

# Morphological Processing
Structure of a word is known as Morhology. A word has some syntax like prefix, root, suffix, thirs person, etc. these are known as morphenes. In this process, we try to train model, that how to form a meaningful word.

# Syntactical Processing
It focuses on how words are arranged to form grammatically correct sentences based on syntactic rules.

# Semantic Processing
Through this, we train model to how to form a meaningful sentence, where to place verb, adjective, etc.
# Discourse processing and contextual processing
Bring the context and try to get true meaning of the sentences. Ex - Put the apple which is currently in the basket onto the shelf, have 2 different meaning, only understandable using "discourse"

# Knowledge base

Till this step,this is known as "NLU - Natural language understanding"

<p> From, Now onward, NLG i.e. Natural Language Generation start</p>

![alt text](<Screenshot from 2025-03-03 17-43-21.png>)

![alt text](<Screenshot from 2025-03-03 17-45-41.png>)

For NLP, we have to use these combination of these:
![alt text](<Screenshot from 2025-03-03 17-46-31.png>)

![alt text](<Screenshot from 2025-03-03 17-48-44.png>)

### Computer Vision
This term refers to the ability of computers to see, Like we humans see, and understans  and process the unstructure stream of information.
![alt text](<Screenshot from 2025-03-04 21-12-34.png>)
For this, there are 3 stages:
# Acquisition 
It means to capture data either in form of photos or videos. Every coloue contains an array of 8 bit integers. So, it helps computer to interpret data.
In this, we do edge detection, segmentation, classification, feature detection. 
# Rendering
generally refers to the process of generating visual content, typically by converting data or models into a visual representation. While the exact meaning can vary depending on the context. In this   
3D Mapping - refers to the generation of 3D images or animations from a model, \  
object recognition - Object recognition is a computer vision task where AI identifies and classifies objects in an image or video. This is typically done using machine learning models, particularly convolutional neural networks (CNNs), which are trained to detect and label various objects. \  
 motion tracking - Motion tracking refers to the process of detecting and following the movement of objects or people across frames in a video. \  
 Auto captioning, Augmented reality, Autonomous Cars, 

 ### Neural networks
 It is like neuron in our brain. It is used for clustering and classify the data. It is also helpful. It is also helpful in extract features which can be fit to other machine learning algorithm under classification and clustering.

 # Perceptron
 It lies under fundamental level of neural network. Its a single algorithm performs binary classification. Basically predicts output is of 1 category or other.
Ex - In this example we will decide, should we take tea or coffee?
![alt text](<Screenshot from 2025-03-05 19-22-34.png>)

<b> Input: <b>   
Let there are 3 inputs

- Weather - cold and rainy = 1 else 0
- Situation - Lots of work = 1 else 0
- Sleep deprivation - 1 else 0


<b> Weights <b>
- 0.3 or 30% to weather
- 0.5 or 50% to work
- 0.2 0r 20% to sleep.

<b> Weighted sum <b>\
 lets say that it is rainy monday after a rather relaxed weekend\
 -> 1*(0.3) + 1*(0.5) + 0*(0.2) = 0.8

<b> Bias <b>\
It is used to force an outcome, influence the acivation function, bringing more flexibility. It can be +ve or -ve.  Here, let  0.4

<b> Activation function <b>\
1, x >=1 and 0, x < 1

<b> Output <b>\
0.8 + 0.4 = 1.2 i.e. 1. So, Drink coffee.


# Single layered Neural networks
![alt text](<Screenshot from 2025-03-05 20-01-00.png>)


 In the above diagram, there are 1 hidden layer with 3 neurons. Each neuron contains All the process from weights to Activation function of perceptron. More number of neurons is helphul because using it we can improve its ability to learn and generalize. Each hidden neuron learns different patterns or features from the input data. It helps our network to capture more complex relationships.

Let we want to train model to identify letter and number. following aare the ways to train model.\
![alt text](<Screenshot from 2025-03-05 21-38-21.png>)   

![alt text](<Screenshot from 2025-03-05 21-38-48.png>)

![alt text](<Screenshot from 2025-03-05 21-39-03.png>)

# Deep Neural Network or Multi Neural Network
A Deep Neural Network (DNN) is a type of artificial neural network that has multiple hidden layers between the input and output layers.
![alt text](<Screenshot from 2025-03-05 20-07-56.png>)

# Feed forward Neural Network
![alt text](<Screenshot from 2025-03-05 22-04-39.png>)
Here, there is multiple hidden layer andno hidden layer is looping. When input enters, it goes directly from left to right, instead of looping between 1 layer to another.

![alt text](<Screenshot from 2025-03-05 22-06-15.png>)
Here, we have 4X4 pixel i.e. 16 pixcel. so, there is 16 input and since, input is number so, here are 10 outputs. Given, number is more similar like a "1". so, 99%, but it is also similar like "7" so, it is 75%.

![alt text](<Screenshot from 2025-03-05 22-14-30.png>)

![alt text](<Screenshot from 2025-03-05 22-15-45.png>)