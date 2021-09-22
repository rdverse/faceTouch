import datetime
import pandas as pd
import numpy as np
import getch

timeStamps = list()
status = list()

fileName = input("Enter file name :")


def reshape_arr(arr):
    arr = np.array(arr)
    arr = arr.reshape(-1, 1)
    return arr


statusMsg = 's'

while statusMsg != 'q':
    print(
        "What is the current status?\n[transition : t, activity : y, disengage : u]\nChoice : "
    )
    statusMsg = getch.getch()
    if statusMsg != 'q':
        status.append(statusMsg)
        dt = datetime.datetime.now()
        timeStamps.append(dt)

    else:
        break

    print(dt, statusMsg)

status = reshape_arr(status)
timeStamps = reshape_arr(timeStamps)

df = pd.DataFrame(np.hstack((timeStamps, status)), columns=['time', 'status'])
print(fileName)
df.to_csv(fileName + '.csv')
