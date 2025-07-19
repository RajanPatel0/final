def transform_input(data, location_columns):
    try:
        sqft = float(data['total_sqft'])
        bhk = int(data['bhk'])
        bath = int(data['bath'])
        balcony = int(data['balcony'])
        location = data['location'].strip().lower()
    except Exception as e:
        print("❌ Error parsing input:", e)
        raise e

    # Start with the 4 numeric features
    features = [sqft, bhk, bath, balcony]

    # Create a zero vector for all location one-hot encodings
    location_vector = [0] * len(location_columns)

    # Match the location (case insensitive)
    for idx, loc in enumerate(location_columns):
        if loc.lower().strip() == location:
            location_vector[idx] = 1
            break

    # Append the one-hot location vector to numeric features
    features.extend(location_vector)

    print("✅ Final features length:", len(features))  # Should print 1202
    return features
