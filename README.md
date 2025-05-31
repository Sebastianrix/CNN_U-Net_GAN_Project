# Comic Colorization Project

## Project Structure
```
├── Data/               # Data directory
│   ├── raw/           # Raw comic images
│   └── prepared_data/ # Preprocessed numpy arrays
├── Models/            # Saved model checkpoints
├── Src/              # Source code
├── Notebooks/        # Jupyter notebooks
└── Results/          # Output colorized images
```

## Setup and Installation

1. Clone the repository
```bash
git clone https://github.com/Sebastianrix/CNN_U-Net_GAN_Project
cd CNN_U-Net_GAN_Project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Cloud Training (cloud.sdu.dk)

1. Upload project to cloud.sdu.dk
2. Configure environment:
```bash
module load cuda/11.8
module load python/3.9
```

3. Start training:
```bash
sbatch train_on_server.sh
```

4. Monitor training:
```bash
squeue -u <username>  # Check job status
tail -f train_output_*.log  # Monitor output
```

## Local Testing

```bash
python Src/test_unet_hsv.py
```

Results will be saved in the `Results/` directory.

## Model Architecture

- U-Net architecture with HSV color space
- Input: Grayscale images (V channel)
- Output: H and S channels
- Loss: Custom HSV loss function with circular hue loss

## License

[Your License]
