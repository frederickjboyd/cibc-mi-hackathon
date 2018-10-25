import pandas as pd


class files:
    # When initializing, read data from a CSV file and store it
    def __init__(self, directory):
        self.data = self.readData(directory)

    def readData(self, directory):
        data = pd.read_csv(directory, header=None)
        # Format dataset as a matrix
        return data

    def printData(self):
        print (self.data)

    def getData(self):
        return self.data
