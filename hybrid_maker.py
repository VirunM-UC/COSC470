import pandas as pd
import utils
for split in ["training", "validation", "testing"]:
    df_paris = pd.read_pickle(f"data-folders/paris-data/{split}.pkl")
    df_global = pd.read_pickle(f"data-folders/material-data/{split}.pkl")
    df_global = df_global.loc[:, ["City_Name", "POINT_X", "POINT_Y", "building_material", "image"]]
    df_paris = df_paris.rename(columns = {"mur": "building_material", "lat": "POINT_Y", "lon": "POINT_X"})
    translation = {
                "beton": "cinder",
                "briques": "brick"
            }
    df_paris["building_material"] = df_paris["building_material"].map(lambda x: translation[x])
    df_paris.insert(0, "City_Name", ["paris" for _ in range(len(df_paris))]) 
    df_paris = df_paris.loc[:, ["City_Name", "POINT_X", "POINT_Y", "building_material", "image"]]
    df = pd.concat([df_global, df_paris], ignore_index = True)

    
    #upsample
    df_brick = df.loc[df["building_material"] == "brick"]
    df_cinder = df.loc[df["building_material"] == "cinder"]
    df_class = [df_brick, df_cinder]
    for i in range(len(df_class)):
        df_class[i].insert(0, "upsample", (df_class[i]["City_Name"] == "paris"))
        df_class[i] = utils.upsample(df_class[i], [False, True], "upsample")
        df_class[i].pop("upsample")
    df = pd.concat(df_class, ignore_index = True)
    
    df = df.sample(frac=1, random_state=utils.RANDOM_STATE) #shuffle
    df.to_pickle(f"data-folders/hybrid-data/{split}.pkl")