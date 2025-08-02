import numpy as np

#data for testing

before=np.full((3,3),8)
after=np.full((3,3),5)

np.save("before_testing.npy",before)
np.save("after_mining.npy",after)

print("DONE")
