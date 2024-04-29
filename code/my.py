import pandas as pd
import os
from shutil import copyfile, copytree, rmtree

datasets = ['mnist','cifar10','reuters','imdb']
model_num = {'blob':39,'circle':36, 'cifar10':35, 'mnist':78, 'reuters':32,'imdb':13}
# res = []
for dataset in datasets:
    for i in range(model_num[dataset]):
        print(i)
        # if not os.path.exists(f"../models/{dataset}"):
        #     os.mkdir(f"../models/{dataset}")
        # if not os.path.exists(f"../models/{dataset}/{i}"):
        #     os.mkdir(f"../models/{dataset}/{i}")
        # copyfile(f"../../DeepFD/CodeBook/Evaluation/{dataset}/{i}/model.h5",f"../models/{dataset}/{i}/model.h5")
        # copyfile(f"../../DeepFD/CodeBook/Evaluation/{dataset}/{i}/training_config.pkl",f"../models/{dataset}/{i}/training_config.pkl")
        # os.rename(f"../models/{dataset}/{i}",f"../models/{dataset}/{c}")
        if os.path.exists(f"../models/{dataset}/{i}/monitor_features"):
            rmtree(f"../models/{dataset}/{i}/monitor_features")
            # os.remove(f"../models/{dataset}/{i}/monitor_features")
        # res.append([dataset,c,i])
        # c+=1
# df = pd.DataFrame(res,columns=['dataset','num','old_num'])
# df.to_csv("data.csv",index=False)