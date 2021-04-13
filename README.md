# Eluvio_DS_challenge
Coding Challenge Option 1

In this repository, I am using the Eluvio_DS_Challenge.csv file to analyze the relation between upvotes and the titles. In particular, my goal is to predict the number of upvotes based on the title of that post using a simple linear regression model. Different from the existing NLP methods, which usually tokenize the input text into numerical values, I propose to convert the string title into images and then load the image pixel values (normalized) as the input. I am using 80% dataset for training and 20% for testing, and using mean squared error as my error metric. The testing result is an MSE of . Future improvement can be added, such as adding the submission time, author name into the training features.      
