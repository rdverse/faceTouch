import datetime
import pandas as pd

timeStamps = list()

while True:
    status  = input()

    if status=='q':
        break

    dt = datetime.datetime.now()
    print(dt)

