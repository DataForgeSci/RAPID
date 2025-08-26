import sys
import os
import re

def get_custom_epochs(inputs_file):
    marker = "28>Input the number of epochs for CNN training:"
    with open(inputs_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith(marker):
            if i + 1 < len(lines):
                next_line = lines[i+1].strip()
                try:
                    return int(next_line)
                except ValueError:
                    return None
    return None

def update_train_script(epoch_count):
    filename = os.path.join("..", "CNN", "train_xrd_basic.py")

    if not os.path.isfile(filename):
        print(f"[Error] Could not find {filename}.")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    old_pattern = r"for\s+epo\s+in\s+range\(\s*10000\s*\)\s*:"
    new_line = f"for epo in range({epoch_count}):"

    new_content, subs_made = re.subn(old_pattern, new_line, content)
    if subs_made == 0:
        print("[Warning] No occurrences of 'range(10000)' found.")
    else:
        print(f"[Info] Updated epoch count to {epoch_count} in {filename}.")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)

def main():
    if len(sys.argv) < 2:
        print("Usage: python update_epochs.py <inputs_file>")
        sys.exit(1)

    inputs_file = os.path.abspath(sys.argv[1])
    if not os.path.isfile(inputs_file):
        print(f"[Error] Could not find the inputs file: {inputs_file}")
        sys.exit(1)

    epoch_count = get_custom_epochs(inputs_file)
    if epoch_count is None:
        print("[Warning] Could not parse epoch count from inputs file. Exiting without changes.")
        return

    update_train_script(epoch_count)

if __name__ == "__main__":
    main()
