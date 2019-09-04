# Calculate Mean Absolute Error
def fitness(predict, actual):
    samples = predict.shape[0]
    mae = 0
    for i in range(samples):
        mae = mae + abs(predict[i] - actual[i])
    mae = mae / samples

    return mae