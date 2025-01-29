import numpy as np
import matplotlib.pyplot as plt
def visualize(loss, y_test, threshold_percent, info, num_epochs, batch_size):


    # Detect anomalies (set a threshold for reconstruction error)
    #threshold = np.percentile(loss, threshold_percent)
    threshold = np.mean(loss) + 2*np.std(loss)
    anomalies = loss > threshold

   
  

    # Example intervals for correct anomalies
    def string_to_intervals(interval_str):
        """Convert a string representation of intervals to a numpy array."""
        return np.array(eval(interval_str))

    true_anomaly_intervals = string_to_intervals(y_test["anomaly_sequences"][0])

   
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Reconstruction loss')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

    # Highlight true anomaly intervals
    for start, end in true_anomaly_intervals:
        plt.axvspan(start, end, color='yellow', alpha=0.3, label='True Anomalies' if start == true_anomaly_intervals[0][0] else "")

    # Highlight anomalies over threshold in red
    anomaly_indices = np.where(anomalies)[0]
    plt.scatter(anomaly_indices, loss[anomaly_indices], color='red', label='Detected Anomalies')

    plt.legend()
    plt.title(f'Reconstruction Loss and Anomaly Detection: {info}')
    plt.xlabel('Sample Index')
    plt.ylabel('Loss')
    plt.savefig(f"plots/{info}_{num_epochs}_{batch_size}_{threshold:.5f}.png", dpi=300)
    plt.show()
    
    
    print(f"Model: {info}")
    print(f"Anomaly detection threshold: {threshold:.5f}")
    print(f"Number of anomalies: {np.sum(anomalies)}")