To train the baseline model: 
save_images.py collects the images using Google APIs, 
create_datasets.py splits the datasets and stores them as pickled Pandas Dataframes, 
The files in attribute-scripts train the model: building_material.py and structure_type.py,
inference.py evaluates the model by city and region.