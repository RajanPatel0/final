def transform_input(data, location_columns):
    import numpy as np
    x = [
        data["total_sqft"],
        data["bath"],
        data["balcony"],
        data["bhk"]
    ]
    # one-hot encode location
    location_vector = [0] * len(location_columns)
    if data["location"] in location_columns:
        index = location_columns.index(data["location"])
        location_vector[index] = 1
    return np.array(x + location_vector)
