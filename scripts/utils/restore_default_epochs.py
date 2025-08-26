import os
import re

def restore_defaults():
    filename = os.path.join("..", "CNN", "train_xrd_basic.py")
    if not os.path.exists(filename):
        print(f"[Warning] Could not find {filename}.")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r"for\s+epo\s+in\s+range\(\s*\d+\s*\)\s*:"
    replacement = "for epo in range(10000):"
    new_content, subs_made = re.subn(pattern, replacement, content)

    if subs_made == 0:
        print("[Warning] No lines changed. Possibly already default or pattern changed.")
    else:
        print("[Info] Restored epoch=10000 in train_xrd_basic.py.")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)

def main():
    restore_defaults()

if __name__ == "__main__":
    main()
