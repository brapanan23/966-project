# Code for running experiments on pixelation levels using MNIST
import os
import csv
import re
from PIL import Image

def parse_pixelation_level(filename):

    match = re.search(r'(\d+\.\d+)_pixelate\.png', filename)
    if match:
        return float(match.group(1))
    return None

def get_image_directories(pixelated_dir='pixelated_mnist'):

    if not os.path.exists(pixelated_dir):
        raise FileNotFoundError(f"Directory {pixelated_dir} not found!")
    
    dirs = []
    for item in os.listdir(pixelated_dir):
        item_path = os.path.join(pixelated_dir, item)
        if os.path.isdir(item_path) and item.startswith('image_'):
            match = re.search(r'image_(\d+)_class_(\d+)', item)
            if match:
                img_idx = int(match.group(1))
                class_label = int(match.group(2))
                dirs.append((img_idx, class_label, item_path))
    
    # Sort by image index
    dirs.sort(key=lambda x: x[0])
    return dirs

def get_pixelation_files(img_dir):

    files = []
    for filename in os.listdir(img_dir):
        if filename.endswith('_pixelate.png'):
            level = parse_pixelation_level(filename)
            if level is not None:
                files.append((level, os.path.join(img_dir, filename)))
    
    files.sort(key=lambda x: x[0], reverse=True)
    return files

def display_image(image_path, pixelation_level, image_idx):

    img = Image.open(image_path)
    img_large = img.resize((img.width * 8, img.height * 8), Image.NEAREST)
    img_large.show()
    print(f"\nImage {image_idx:03d} | Pixelation Level: {pixelation_level}")

def get_user_name():

    while True:
        name = input("Please enter your name: ").strip()
        if name:
            # Sanitize name for filesystem (remove invalid characters)
            name = re.sub(r'[<>:"/\\|?*]', '_', name)
            return name
        print("Name cannot be empty. Please try again.")

def run_experiment(pixelated_dir='pixelated_mnist', user_name=None, 
                   start_from_image=None, resume=False):

    if user_name is None:
        user_name = get_user_name()
    
    # Create directory structure: human_data/{name}/
    user_data_dir = os.path.join('human_data', user_name)
    os.makedirs(user_data_dir, exist_ok=True)
    output_csv = os.path.join(user_data_dir, 'results.csv')
    
    print(f"\nResults will be saved to: {output_csv}\n")
    
    # Get all image directories
    image_dirs = get_image_directories(pixelated_dir)
    print(f"Found {len(image_dirs)} images to process")
    
    # Load existing results if resuming
    completed_images = set()
    if resume and os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_images.add(int(row['image_idx']))
        print(f"Resuming: {len(completed_images)} images already completed")
    
    # Determine starting point
    start_idx = 0
    if start_from_image is not None:
        start_idx = next((i for i, (idx, _, _) in enumerate(image_dirs) if idx == start_from_image), 0)
    
    # Open CSV file for writing
    file_exists = os.path.exists(output_csv) and not resume
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['image_idx', 'class_label', 'pixelation_level', 
                     'user_guess', 'correct', 'action_taken']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Process each image
        for img_idx, class_label, img_dir in image_dirs[start_idx:]:
            # Skip if already completed
            if resume and img_idx in completed_images:
                continue
            
            # Get pixelation files sorted by level (highest first)
            pixelation_files = get_pixelation_files(img_dir)
            if not pixelation_files:
                print(f"Warning: No pixelation files found in {img_dir}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Image {img_idx:03d}")
            print(f"{'='*60}")
            
            # Start from highest pixelation level
            current_level_idx = 0
            guessed = False
            
            while current_level_idx < len(pixelation_files):
                pixelation_level, image_path = pixelation_files[current_level_idx]
                
                # Display image (don't show true class to user)
                display_image(image_path, pixelation_level, img_idx)
                
                # Get user input
                if current_level_idx < len(pixelation_files) - 1:
                    next_level = pixelation_files[current_level_idx + 1][0]
                    print(f"Options:")
                    print(f"  [0-9] - Guess the digit")
                    print(f"  [n]   - Next level (go to {next_level})")
                    print(f"  [q]   - Quit experiment")
                else:
                    print(f"Options:")
                    print(f"  [0-9] - Guess the digit (this is the lowest pixelation level)")
                    print(f"  [q]   - Quit experiment")
                
                user_input = input("\nYour choice: ").strip().lower()
                
                if user_input == 'q':
                    print("\nExperiment stopped by user.")
                    return
                
                elif user_input == 'n':
                    if current_level_idx < len(pixelation_files) - 1:
                        # Record that user went to next level
                        writer.writerow({
                            'image_idx': img_idx,
                            'class_label': class_label,
                            'pixelation_level': pixelation_level,
                            'user_guess': None,
                            'correct': None,
                            'action_taken': 'next_level'
                        })
                        csvfile.flush()
                        current_level_idx += 1
                    else:
                        print("Already at the lowest pixelation level!")
                
                elif user_input.isdigit() and 0 <= int(user_input) <= 9:
                    guess = int(user_input)
                    correct = (guess == class_label)
                    
                    # Record the guess
                    writer.writerow({
                        'image_idx': img_idx,
                        'class_label': class_label,
                        'pixelation_level': pixelation_level,
                        'user_guess': guess,
                        'correct': correct,
                        'action_taken': 'guess'
                    })
                    csvfile.flush()
                    
                    # Show result
                    if correct:
                        print(f"\n✓ Correct! The digit is {class_label}")
                    else:
                        print(f"\n✗ Incorrect. You guessed {guess}, but the digit is {class_label}")
                    
                    guessed = True
                    break
                
                else:
                    print("Invalid input. Please enter a digit (0-9), 'n' for next level, or 'q' to quit.")
                    continue
            
            # If user went through all levels without guessing, record that
            if not guessed:
                print(f"\nReached lowest pixelation level without a guess.")
                writer.writerow({
                    'image_idx': img_idx,
                    'class_label': class_label,
                    'pixelation_level': pixelation_files[-1][0],
                    'user_guess': None,
                    'correct': None,
                    'action_taken': 'no_guess'
                })
                csvfile.flush()
            
            # Small pause before next image
            print(f"\nMoving to next image...")
    
    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pixelation experiment on MNIST images')
    parser.add_argument('--pixelated_dir', type=str, default='pixelated_mnist',
                       help='Directory containing pixelated images (default: pixelated_mnist)')
    parser.add_argument('--user_name', type=str, default=None,
                       help='User name (if not provided, will prompt)')
    parser.add_argument('--start_from', type=int, default=None,
                       help='Start from specific image index')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from where you left off (skip completed images)')
    
    args = parser.parse_args()
    
    run_experiment(
        pixelated_dir=args.pixelated_dir,
        user_name=args.user_name,
        start_from_image=args.start_from,
        resume=args.resume
    )
