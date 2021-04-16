# Eluvio_DS_challenge
Coding Challenge Option 1

In this repository, I am using the provided Eluvio_DS_Challenge.csv file to analyze the relation between upvotes and the titles. In particular, my goal is to predict the number of upvotes based on the title of that post through a simple linear regression model. 

Different from the existing NLP methods, which usually tokenize the input text into numerical values, I propose to convert the string title into images (shown in the following images) and then load the image pixel values (normalized) as the input. The benefit of doing that is no tokenizer packages are required given special characters may exist in the title.

![image_74671](https://user-images.githubusercontent.com/75220576/114967909-eb6ccd00-9e3a-11eb-87f6-94fdf887c75f.png)
![image_330380](https://user-images.githubusercontent.com/75220576/114967918-ee67bd80-9e3a-11eb-890a-f2b7864730a1.png)

In this repository, the sklearn tool is implemented to train a simple linear regression model. 80% dataset are used for training and 20% for testing. Mean squared error is used as my error metric. The final testing result is bad. Potential reasons are list as follows:

1. The SGDRegressor Model does not fit our dataset.
2. The model may be underfitted as both training acc and testing acc are low.
3. The training sample's feature dimension is not appropriate. In our experiment, its dimension is 6000 (i.e., 80*75).
4. The title is not the only factor that determines the upvotes.    


Further improvement can be made by (1) including the author and submission time, (2) introducing more advance machine learning tools, say deep neural network, to approximate the nonlinear relation between upvotes and titles, (3) convolutional neural network can be implemented to handle the image input.
