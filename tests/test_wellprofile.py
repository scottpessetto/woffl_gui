# import matplotlib.pyplot as plt
# import numpy as np

from woffl.geometry.wellprofile import WellProfile

# only works if the command python -m tests.wprof_test is used

# define a well profile, check that the interpolation works
# how to check the filtration works?

sch_profile = WellProfile.schrader()
sch_profile.plot_raw()
sch_profile.plot_filter()

kup_profile = WellProfile.kuparuk()
kup_profile.plot_raw()
kup_profile.plot_filter()
