import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

df = pd.read_csv('test.csv', sep=',')
print(df.head(5))