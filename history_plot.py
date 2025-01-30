import matplotlib.pyplot as plt
import seaborn as sns

rounds = [1, 2, 3, 4, 5]  # 原始round 0-4
rmse_lstm = [2.5810126066207886, 0.97413769364357, 0.2896924912929535, 0.19316484034061432, 0.15590141713619232] #lstm
rmse_bilstm = [2.6618142127990723, 0.43725696206092834, 0.1569969579577446, 0.1333986446261406, 0.11677738279104233] #bilstm

plt.figure(figsize=(10, 6))

plt.plot(rounds, rmse_lstm, 'o-', color='#1f77b4', linewidth=2, label='LSTM', markersize=8)
plt.plot(rounds, rmse_bilstm, 's-', color='#2ca02c', linewidth=2, label='BiLSTM', markersize=8)

plt.title('LSTM vs BiLSTM Aggregated Validation RMSE - OpenFL', fontsize=14, pad=15)
plt.xlabel('Training Round', fontsize=12)
plt.ylabel('RMSE', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 3.0)
plt.xticks(rounds)

plt.legend(fontsize=10)

plt.tight_layout()
plt.show()