# 966-project
Modeling Visual Inference Under Uncertainty

For this project, we use MNIST data with each image having a set of pixelation levels applied to it.
These can be modified in `src/apply_pixel_levels.py`.

## Directory Structure

```
966-project/
├── src/                          # Source code modules
│   ├── apply_pixel_levels.py    # Applies pixelation to MNIST images
│   │                            # - Creates pixelated versions at different levels
│   │                            # - Saves images to pixelated_mnist/ directory
│   │
│   ├── run_experiment.py        # Interactive GUI experiment runner
│   │                            # - Displays pixelated images to users
│   │                            # - Collects user guesses via keyboard input
│   │                            # - Saves results to human_data/{user_name}/results.csv
│   │
│   ├── get_mnist.py             # MNIST data loading utilities
│   │                            # - get_mnist_raw(): Loads from raw IDX files
│   │                            # - get_mnist_torch(): Loads via torchvision
│   │
│   └── get_emnist.py            # EMNIST data loading utilities
│                                # - Similar functions for EMNIST dataset
│
├── notebooks/                  
│   ├── EDA_mnist.ipynb     
│   │
│   └── EDA.ipynb                
│
├── pixelated_mnist/             # Generated pixelated images
│   └── image_XXX_class_Y/      # One directory per image
│       ├── 01.0_pixelate.png   # Least pixelated
│       ├── 01.2_pixelate.png  
│       ├── ...                  # Intermediate levels
│       └── 06.5_pixelate.png   # Most pixelated
│
├── human_data/                  # User experiment results
│   └── {user_name}/            # One directory per user
│       └── results.csv          # CSV with columns:
│                                #   - image_idx, class_label, pixelation_level
│                                #   - user_guess, correct, action_taken
```

## Key Files

### `src/apply_pixel_levels.py`
**Purpose**: Generate pixelated versions of MNIST images at multiple levels.

**Usage**:
```bash
python src/apply_pixel_levels.py
```

### `src/run_experiment.py`
**Purpose**: Interactive GUI experiment for collecting human perception data.

**Usage**:
```bash
python src/run_experiment.py
# Will prompt for user name, or use:
python src/run_experiment.py --user_name "Alice"
```

### `src/get_mnist.py` & `src/get_emnist.py`
**Purpose**: Data loading utilities for MNIST and EMNIST datasets.