import numpy as np
import matplotlib.pyplot as plt
from unet_model_lab import rgb_to_lab
import os

def analyze_color_distribution():
    print("Loading training data...")
    data_dir = os.path.join('Data', 'prepared_data')
    y_train = np.load(os.path.join(data_dir, "comic_output_color_train.npy"))
    
    print("Converting to LAB color space...")
    lab_images = np.array([rgb_to_lab(img) for img in y_train])
    
    # Analyze L channel
    l_channel = lab_images[:, :, :, 0]
    print("\nL channel statistics:")
    print(f"Range: [{l_channel.min():.2f}, {l_channel.max():.2f}]")
    print(f"Mean: {l_channel.mean():.2f}")
    print(f"Std: {l_channel.std():.2f}")
    
    # Analyze A channel
    a_channel = lab_images[:, :, :, 1]
    print("\nA channel statistics:")
    print(f"Range: [{a_channel.min():.2f}, {a_channel.max():.2f}]")
    print(f"Mean: {a_channel.mean():.2f}")
    print(f"Std: {a_channel.std():.2f}")
    
    # Analyze B channel
    b_channel = lab_images[:, :, :, 2]
    print("\nB channel statistics:")
    print(f"Range: [{b_channel.min():.2f}, {b_channel.max():.2f}]")
    print(f"Mean: {b_channel.mean():.2f}")
    print(f"Std: {b_channel.std():.2f}")
    
    # Plot histograms
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.hist(l_channel.flatten(), bins=50, density=True)
    ax1.set_title('L Channel Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    
    ax2.hist(a_channel.flatten(), bins=50, density=True)
    ax2.set_title('A Channel Distribution')
    ax2.set_xlabel('Value')
    
    ax3.hist(b_channel.flatten(), bins=50, density=True)
    ax3.set_title('B Channel Distribution')
    ax3.set_xlabel('Value')
    
    plt.tight_layout()
    plt.savefig('color_distribution.png')
    plt.close()
    
    # Calculate percentiles for better understanding of value distribution
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    
    print("\nPercentile analysis:")
    print("\nL channel percentiles:")
    for p in percentiles:
        value = np.percentile(l_channel, p)
        print(f"{p}th percentile: {value:.2f}")
        
    print("\nA channel percentiles:")
    for p in percentiles:
        value = np.percentile(a_channel, p)
        print(f"{p}th percentile: {value:.2f}")
        
    print("\nB channel percentiles:")
    for p in percentiles:
        value = np.percentile(b_channel, p)
        print(f"{p}th percentile: {value:.2f}")

if __name__ == "__main__":
    analyze_color_distribution() 