import pandas as pd

df_paris = pd.read_pickle("data-folders/paris-data/validation.pkl")
df_global = pd.read_pickle("data-folders/fov90-data/validation.pkl")
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
df.to_pickle("data-folders/hybrid-data/validation.pkl")