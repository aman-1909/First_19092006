import numpy as np

def detect_mining(before,after,minimum=0.5) :
 change = before-after
 mask=change>minimum
 volume=np.sum(change[mask])
 return change , mask,volume
