import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

debug = True
visuals = True


def main():
    startDate = datetime.datetime(2017, 1, 3)
    endDate = datetime.datetime(2018, 12, 31)

    # TRAIN DATA
    # Obtain train Data
    trainDF = pd.read_csv("train.csv")

    # Convert columns to right data types
    trainDF['Date'] = pd.to_datetime(trainDF['Date'])
    trainDF['Date'] = list((range(1, 503)))
    trainDF['Volume'] = trainDF['Volume'].apply(lambda v: float(str(v)[:-1]))

    print("Train data Obtained")
    if debug:
        print(str(trainDF.info()) + "\n")
        print("Columns: " + str(list(trainDF.columns)) + "\n")
        print(str(trainDF.head(5)) + "\n")
        print("Check nulls: \n" + str(trainDF.isnull().any()) + "\n\n\n")

    # Separate Attributes/Features and Labels (closing prices)
    # Enumerated Date and used it
    X = trainDF[['Date', 'Open', 'High', 'Low', 'Adj;Close', 'Volume']].values
    Y = trainDF[['Close']].values

    # Check for outliers/discrepancies, go over data
    if debug:
        print("\n Train Data stats: ")
        print(str(trainDF.describe()) + "\n\n\n\n")

    # GET DATA: DONE

    # PLOT TRAIN DATA
    # Plot Box and Whisker Plot
    grph = trainDF[['Date', 'Open', 'High', 'Low', 'Close', 'Adj;Close']].plot(kind='box',
                                                                               title='Distribution of Train data')
    grph.set_ylabel('Price ($)')
    if visuals:
        plt.show()

    # Plot Volume separately because it is so much bigger than the rest
    grph = trainDF[['Volume']].plot(kind='box', title="Volume (Training data)")
    grph.set_ylabel('Stocks Exchanged (#e7)')
    if visuals:
        plt.show()

    # Plot Average Close Price
    plt.figure(figsize=(12, 6))
    plt.tight_layout()
    grph = sb.distplot(trainDF['Close']).set_title("Average Closing Price")
    plt.xlabel("Closing ($)")
    if visuals:
        plt.show()

    # PLOT TRAIN DATA DONE

    # INTERNAL TESTING
    # Split data into 80/20 train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    print("Internal Testing: Data split")
    if debug:
        print("X_train len: " + str(len(X_train)))
        print("X_test len: " + str(len(X_test)) + "\n")

    # Create and Train our model
    lreg = LinearRegression()
    fit = lreg.fit(X_train, Y_train)
    if debug:
        print("Linear reg intercept: " + str(lreg.intercept_))
        print("Linear reg slope: " + str(lreg.coef_) + "\n")

    # Predict Y axis
    y_predict = lreg.predict(X_test)

    # Plot line of regression
    if debug:
        print("X_test len: " + str(len(X_test)))
        print("y_predict len: " + str(len(y_predict)))

    # Compare Prediction with Actual Values
    df = pd.DataFrame({'Actual': list(Y_test), 'Predicted': list(y_predict)})
    df.index += 1
    df['Actual'] = df['Actual'].astype(float)
    df['Predicted'] = df['Predicted'].astype(float)
    print(df.head(10))

    # Plot Internal testing results
    grph = df.head(20).plot(kind='bar', figsize=(10, 6), title='Internal Testing')
    grph.set_ylabel('Closing Price ($)')
    grph.set_xlabel('Date')
    plt.grid(which='major', linestyle=':', linewidth='0.4', color='purple')
    if visuals:
        plt.show()

    # Display internal testing stats
    if debug:
        print('\nMean Absolute Error:', metrics.mean_absolute_error(Y_test, y_predict))
        print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_predict))
        print('Root Mean Squared Error:', str(np.sqrt(metrics.mean_squared_error(Y_test, y_predict))) + "\n\n\n\n")

    # INTERNAL TESTING: DONE

    # TEST DATA
    testDF = pd.read_csv("test.csv")

    # Make columns correct type
    testDF['Date'] = pd.to_datetime(testDF['Date'])
    testDF['Date'] = list((range(503, 755)))
    testDF['Volume'] = testDF['Volume'].apply(lambda v: float(str(v)[:-1]))

    print("Test Data obtained")
    if debug:
        print(str(testDF.info()) + "\n")
        print("Columns: " + str(list(testDF.columns)) + "\n")
        print(str(testDF.head(5)) + "\n")
        print("Check nulls: \n" + str(trainDF.isnull().any()) + "\n\n\n")

    # Split train data into X_train, Y_train, X_test, Y_test
    X_train = trainDF[['Date', 'Open', 'High', 'Low', 'Adj;Close', 'Volume']]
    Y_train = trainDF['Close']
    X_test = testDF[['Date', 'Open', 'High', 'Low', 'Adj;Close', 'Volume']]
    Y_test = testDF['Close']

    # Fit/Train model to new data
    fit = lreg.fit(X_train, Y_train)
    if debug:
        print("Linear reg intercept: " + str(lreg.intercept_))
        print("Linear reg slope: " + str(lreg.coef_) + "\n\n\n")

    # Predict Y data (Closing prices)
    y_predict = lreg.predict(X_test)
    if debug:
        print("Below Should be same length")
        print("X_test len: " + str(len(X_test)))
        print("y_predict len: " + str(len(y_predict)) + "\n\n\n")

    # Compare Prediction with Actual Values
    df = pd.DataFrame({'Actual': list(Y_test), 'Predicted': list(y_predict)})
    df.index += 503
    df['Actual'] = df['Actual'].astype(float)
    df['Predicted'] = df['Predicted'].astype(float)
    print(df.head(10))

    # WRITE RESULTS
    # Write Actual y values to Actual.txt
    with open("Actual.txt", "w") as f:
        for y in testDF['Close']:
            f.write(str(y) + "\n")
    # Write predicted y values to Results.txt
    with open("Results.txt", "w") as f:
        for y in y_predict:
            f.write(str(y) + "\n")
    # Write Correct or Wrong depending on prediction accuracy
    with open("info.txt", "w") as f:
        for x, y in zip(df['Predicted'], df['Actual']):
            if round(x, 4) == round(y, 4):
                f.write("Correct\n")
            else:
                f.write("Wrong!\n")


if __name__ == "__main__":
    main()

# Train data has 501 lines
# Test Data has 251 lines

# Ideas
# Array full of features
