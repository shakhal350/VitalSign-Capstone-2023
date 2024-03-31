import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mvn_model_def import ViTForHeartRateEstimation, MAEModel  
import matplotlib.pyplot as plt

# new data is in similar in format of training data
def prepare_new_data(file_path):
    df = pd.read_csv(file_path)
    
    new_phase_data = df.iloc[:, 1].values  # phase values in second column
    actual_heart_rates = df.iloc[:, 2].values
    scaler = StandardScaler()
    new_phase_data_scaled = scaler.fit_transform(new_phase_data.reshape(-1, 1))  # Reshape for single feature scaling
    
    new_data_tensor = torch.FloatTensor(new_phase_data_scaled)
    actual_hr_tensor = torch.FloatTensor(actual_heart_rates)
    
    return new_data_tensor, actual_hr_tensor

def test_model(model_path, new_data_tensor, actual_hr_tensor):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae_model = MAEModel().to(device)  # Initialize MAE model
    vit_model = ViTForHeartRateEstimation(mae_model).to(device)  # Initialize ViT model
    vit_model.load_state_dict(torch.load(model_path))
    vit_model.eval()  # Set the model to evaluation mode

    # Create DataLoader for new data
    new_data_loader = DataLoader(TensorDataset(new_data_tensor), batch_size=32)

    # Predict
    predictions = []
    with torch.no_grad():
        for batch in new_data_loader:
            inputs = batch[0].to(device)
            outputs = vit_model(inputs).squeeze(1)  # Adjust based on your output
            predictions.extend(outputs.cpu().numpy())

        # After prediction loop
    actual_hr_np = actual_hr_tensor.numpy()
    
    # Plot predictions against actual heart rates
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predicted Heart Rates', marker='o', linestyle='-', color='b')
    plt.plot(actual_hr_np, label='Actual Heart Rates', marker='x', linestyle='--', color='r')
    plt.title('Predicted vs Actual Heart Rates')
    plt.xlabel('Sample')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate and print some error metrics, e.g., MSE
    mse = ((predictions - actual_hr_np) ** 2).mean()
    print(f"Mean Squared Error: {mse}")

    return predictions, actual_hr_np

# Update usage
model_path = r"filepath"
new_data_path = r"filepath"
new_data_tensor, actual_hr_tensor = prepare_new_data(new_data_path)
predictions, actual_hr_np = test_model(model_path, new_data_tensor, actual_hr_tensor)

# You can still save the DataFrame of predictions if you want to
predictions_df = pd.DataFrame({
    'Predicted Heart Rate': predictions,
    'Actual Heart Rate': actual_hr_np
})
print(predictions_df)
# Save predictions if needed
# predictions_df.to_csv(r"C:\Users\Grace\Downloads\new_data_predictions_39.csv", index=False)
