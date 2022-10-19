import matplotlib.pyplot as plt
import pandas as pd

be = pd.read_csv(r'csv/bannerelk.csv')
vc = pd.read_csv(r'csv/vallecrucis.csv')
bth = pd.read_csv(r'csv/bethel.csv')
wat = pd.read_csv(r'csv/watauga.csv')

dates = pd.to_datetime(wat["datetime"], format="%Y%m%d")


#print(be)
#print(vc)
#print(bth)
#print(wat)

fig,ax = plt.subplots(5, figsize = (18,9))

ax[0].plot(dates, wat["level"])
ax[0].set_title("Watauga Level")
ax[0].set_ylabel("CFS")

ax[1].plot(dates, wat["precip"])
ax[1].set_title("Watauga Gauge Precip")
ax[1].set_ylabel("inches")

ax[2].plot(dates, be["precip"])
ax[2].set_title("Banner Elk Precip")
ax[2].set_ylabel("inches")

ax[3].plot(dates, vc["precip"])
ax[3].set_title("Valle Crucis Precip")
ax[3].set_ylabel("inches")

ax[4].plot(dates, bth["precip"])
ax[4].set_title("Bethel Precip")
ax[4].set_ylabel("inches")

plt.tight_layout()
plt.show()
