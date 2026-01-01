# %%
from cityseer import rustalgos

test = rustalgos.seconds_from_distances([80, 400, 800, 1200, 1600, 4800], 1.33333)
test = [t / 60 for t in test]
test

# %%
