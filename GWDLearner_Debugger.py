import os
import pandas as pd
from GWDLearner import *


DATA_PATH = "./data/sim_data_yi/"

users_df   = pd.read_csv(DATA_PATH + "users_profile.csv", header = 0, sep = ",")
friends_df = pd.read_csv(DATA_PATH + "friendships.csv", header = 0, sep = ",")
dist_df    = pd.read_csv(DATA_PATH + "dist_mat.csv", header = 0, sep = ",")

friends_df = friends_df[friends_df.isFriend == 1]
friends_df["pair"] = friends_df[["uid_a", "uid_b"]].apply(lambda x: (int(x[0]), int(x[1])), axis=1)
friends_df.drop("isFriend", axis=1, inplace=True)
friends_df = friends_df[["pair", "uid_a", "uid_b"]]
friends_df.head(3)

cols = ["x0", "x1", "x2", "x3", "x4", "x5"]

## subset users data to retain profile only
profile_df = users_df[["ID"] + cols]
all_user_ids = list(set(users_df.ID))

## ###################################################
## start learning
profile_df = profile_df      # user profile
friends_ls = friends_df.pair # user relationship

res = learning_wrapper(profile_df=profile_df, friends_pair=friends_ls, max_iter=5,
                       k=2, c=0.1, dropout_rate=0.2, fit_rayleigh=True, verbose=True)
## #################################
## Export results
import json

root_path = os.getcwd()
data_path = root_path + "/results/"

# extract component of interest
_, _, info_pkg = res

# create output connection
outfile = data_path + "beta_test_20150822_v01.json"
out_conn = open(outfile, 'w')
print "Writing out information..."
out_conn.write( json.dumps(info_pkg) )
out_conn.close()