import json
import numpy as np

with open("golden_meshes/session_20260104_132429/canonical/FRONTAL_2D.json") as f:
    data = json.load(f)

arr = np.array(data["canonical_2d"])
print(arr.shape)
