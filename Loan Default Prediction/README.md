# Dataset Source - 
https://www.kaggle.com/datasets/nikhil1e9/loan-default

# About the dataset - 
Different banks provide different kinds of loan to different kinds of people. And no bank in the world would want to have the loan payments defaults from their client which is bad for business. Hence this dataset comprises of the information financial institution have about their clients including but not limited to their age, education, marital status, their loan amount and interest rates and harness the power of machine leanring to figure out, which of their clients are more likely to default their loan payments so that required actions can be taken to minimise the company loses.

# Personal goal for the dataset -
I have used this dataset as an experiment to figure out how a LightGBM Classifer algorithm work in terms of predicting the default cases where you don't even have to convert the categorical data into numerical format as this problem is taken care by the LightGBM algorithm itself. And compared the result with other classifer algorithms such as Random Forest classifer, XGB Classifier, Logistic Regression classifer, and KNeighbors Classifer.

Fortunately for me, this dataset was very clean and i didn't have to do a lot of data preprocessing except changing the categorical features to numerical features before feeding it to the ML algorthms.

# Results: 
To my surprise the LightGBM algorithm worked relatively well against KNeighbors classifier but the random forest and XGB classifer showed better results with 89% accuracy compared to LighGBM classifer with 88% accuracy for this dataset.

# Feedback: 
I am always looking forward towards a feedback on my work, do reach out if you think I could have done certain thing in a more efficient way and help me towards my journey to get better at machine learning exploration.
