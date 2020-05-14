# Stock-Prediction
Utilizes 2017-2018 Tesla stock information to predict closing Tesla stock for 2019

## Dataset
- train - 2017 to 2018 Tesla stock information 
- test - 2019 Tesla stock information (no closing price)

## Approach
Obtain the full set of data 2017 to 2019, split into training and test. Take out closing stock price for test because that is what we want
to predict. Create our multivariable linear regression model using features: Date, Open, High, Low, Adj;Close, Volume to predict 
our label, Closing price. Program will output 3 files after finished, Actual.txt, Results.txt and info.txt

###### Actual.txt
Real Closing prices for Tesla stock 2018

###### Results.txt
Predicted Closing prices for Tesla stock 2018

###### info.txt
Correct if predicted maches real
Incorrect! if it does not match
