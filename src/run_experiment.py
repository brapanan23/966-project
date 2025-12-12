# Code for running experiments on pixelation levels using MNIST
import os
import csv
import re
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

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

def show_image_gui(image_path, image_idx, level_num, total_levels, has_next_level):
    """
    Display image in a GUI window with keyboard-only input.
    Returns: 'guess' with digit (0-9), 'next', or 'quit'
    
    Args:
        image_path: Path to the image file
        image_idx: Image index number
        level_num: Current level number (1 = most pixelated, total_levels = least pixelated)
        total_levels: Total number of pixelation levels
        has_next_level: Whether there's a next (less pixelated) level available
    """
    result = {'action': None, 'guess': None}
    
    root = tk.Tk()
    root.title(f"MNIST Experiment - Image {image_idx:03d}")
    root.resizable(False, False)
    
    # Make window focusable for keyboard input
    root.focus_set()
    root.focus_force()
    
    # Load and resize image
    img = Image.open(image_path)
    img_large = img.resize((img.width * 12, img.height * 12), Image.NEAREST)
    photo = ImageTk.PhotoImage(img_large)
    
    # Create main frame
    main_frame = tk.Frame(root, padx=40, pady=30)
    main_frame.pack()
    
    # Title
    title_label = tk.Label(main_frame, text=f"Image {image_idx:03d} | Level {level_num} of {total_levels}", 
                           font=('Arial', 18, 'bold'))
    title_label.pack(pady=(0, 20))
    
    # Image display
    image_label = tk.Label(main_frame, image=photo)
    image_label.pack(pady=20)
    
    # Instructions - more prominent
    instruction_lines = []
    instruction_lines.append("Press a digit key (0-9) to guess")
    if has_next_level:
        instruction_lines.append(f"Press 'n' for next level (Level {level_num + 1} of {total_levels})")
    else:
        instruction_lines.append(f"(This is Level {level_num} - the least pixelated)")
    instruction_lines.append("Press 'q' to quit")
    
    for i, line in enumerate(instruction_lines):
        instruction_label = tk.Label(main_frame, text=line, font=('Arial', 14))
        if i == 0:
            instruction_label.pack(pady=(20, 8))
        else:
            instruction_label.pack(pady=4)
    
    # Keyboard shortcuts
    def on_key_press(event):
        key = event.char.lower()
        if key.isdigit() and 0 <= int(key) <= 9:
            on_guess(int(key), root, result)
        elif key == 'n' and has_next_level:
            on_next(root, result)
        elif key == 'q':
            on_quit(root, result)
    
    root.bind('<Key>', on_key_press)
    
    # Keep reference to photo to prevent garbage collection
    image_label.image = photo
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Wait for user action
    root.mainloop()
    
    return result['action'], result['guess']

def on_guess(digit, root, result):
    """Handle digit guess"""
    result['action'] = 'guess'
    result['guess'] = digit
    root.quit()
    root.destroy()

def on_next(root, result):
    """Handle next level"""
    result['action'] = 'next'
    root.quit()
    root.destroy()

def on_quit(root, result):
    """Handle quit"""
    if messagebox.askyesno("Quit", "Are you sure you want to quit the experiment?"):
        result['action'] = 'quit'
        root.quit()
        root.destroy()

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
    
    csv_exists = os.path.exists(output_csv)
    
    # Load existing results to determine completed images
    completed_images = set()
    if csv_exists:
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    img_idx_str = row.get('image_idx', '').strip()
                    # Skip header row or empty rows
                    if not img_idx_str or img_idx_str == 'image_idx' or not img_idx_str.isdigit():
                        continue
                    img_idx = int(img_idx_str)
                    action = row.get('action_taken', '').strip()
                    # Only mark as completed if user made a guess or reached lowest level without guessing
                    if action in ['guess', 'no_guess']:
                        completed_images.add(img_idx)
                except (ValueError, KeyError) as e:
                    # Skip malformed rows
                    continue
    
    # Auto-enable resume if there are completed images (unless explicitly disabled)
    if csv_exists and completed_images and not resume:
        resume = True
        print(f"\nFound existing results file with {len(completed_images)} completed images.")
        print("Resuming from where you left off.\n")
    
    print(f"Results will be saved to: {output_csv}\n")
    
    # Get all image directories
    image_dirs = get_image_directories(pixelated_dir)
    print(f"Found {len(image_dirs)} images to process")
    
    if resume and completed_images:
        print(f"Skipping {len(completed_images)} already completed images.")
    
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
            
            total_levels = len(pixelation_files)
            
            while current_level_idx < len(pixelation_files):
                pixelation_level, image_path = pixelation_files[current_level_idx]
                
                # Calculate level number (1 = most pixelated, total_levels = least pixelated)
                level_num = current_level_idx + 1
                
                # Check if there's a next level
                has_next = current_level_idx < len(pixelation_files) - 1
                
                # Show GUI and get user input
                action, guess = show_image_gui(image_path, img_idx, level_num, total_levels, has_next)
                
                if action == 'quit':
                    print("\nExperiment stopped by user.")
                    return
                
                elif action == 'next':
                    if has_next:
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
                
                elif action == 'guess':
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
                    
                    # Show result in a message box
                    if correct:
                        messagebox.showinfo("Result", f"✓ Correct! The digit is {class_label}")
                    else:
                        messagebox.showinfo("Result", f"✗ Incorrect. You guessed {guess}, but the digit is {class_label}")
                    
                    guessed = True
                    break
            
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
