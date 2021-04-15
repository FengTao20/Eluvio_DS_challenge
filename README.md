# Eluvio_DS_challenge
Coding Challenge Option 1

In this repository, I am using the provided Eluvio_DS_Challenge.csv file to analyze the relation between upvotes and the titles. In particular, my goal is to predict the number of upvotes based on the title of that post through a simple linear regression model. Different from the existing NLP methods, which usually tokenize the input text into numerical values, I propose to convert the string title into images and then load the image pixel values (normalized) as the input. I am using 90% dataset for training and 10% for testing, and using mean squared error as my error metric. The testing result is an MSE of . Future improvement can be added, such as adding the submission time, author name into the training features.      
