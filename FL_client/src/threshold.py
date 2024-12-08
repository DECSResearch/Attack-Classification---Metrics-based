import numpy as np
import joblib

model = joblib.load('trained_model_LSTM.joblib')

# Calculate MAE for training prediction
trainPredict = model.predict(trainX)
trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)

# Print the mean of test MAE
print("Mean of Train MAE:", np.mean(trainMAE))

# Calculate MAPE for each sample
trainActual = trainX
trainMAPE = np.mean(np.abs(trainPredict - trainActual) / trainActual, axis=1) * 100

# Print the mean of MAPE
print("Mean of Train MAPE:", np.mean(trainMAPE))

# Print shapes of MAE and MAPE
print(trainMAE.shape)
print(trainMAPE.shape)

# Percentile Method to define anomaly thresholds
thresholds = []
percentile = 95

for i in range(trainMAPE.shape[1]):
    threshold = np.percentile(trainMAPE[:, i], percentile)
    thresholds.append(threshold)
    print(f"Training Feature: {feature_names[i]}, Threshold ({percentile} Percentile): {threshold:.2f}")


# Calculate reconstruction loss (MAE) for testing dataset
testPredict = model.predict(X_test)
testMAE = np.mean(np.abs(testPredict - X_test), axis=1)

# Print the mean of test MAE
print("Mean of Test MAE:", np.mean(testMAE))

# Calculate MAPE for each sample
testActual = X_test  # Assuming trainX contains the actual values
testMAPE = np.mean(np.abs(testPredict - testActual) / testActual, axis=1) * 100

# Print the mean of MAPE
print("Mean of Test MAPE:", np.mean(testMAPE))

# Percentile Method to define anomaly thresholds
thresholds = []
percentile = 95

for i in range(testMAPE.shape[1]):
    threshold = np.percentile(testMAPE[:, i], percentile)
    thresholds.append(threshold)
    print(f"Test Feature: {feature_names[i]}, Threshold ({percentile} Percentile): {threshold:.2f}")