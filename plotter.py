import matplotlib.pyplot as plt



file = open("./sourceData/train.txt")
speedTruthArrayString = file.readlines()
speedTruthArray = []
for numeric_string in speedTruthArrayString:
    numeric_string = numeric_string.strip('\n')
    speedTruthArray.append(float(numeric_string))
file.close()


plt.plot(speedTruthArray, label = "Ground truth speeds")
# naming the x axis
plt.xlabel('Frame')
# naming the y axis
plt.ylabel('Speed')
# giving a title to my graph
plt.title('Speed per frame chart')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()