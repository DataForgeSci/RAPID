import os
import re
import matplotlib.pyplot as plt

def parse_cnn_results(file_path):
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('train acc/loss'):
                parts = line.strip().split()
                acc = float(parts[2])
                loss = float(parts[3])
                train_acc.append(acc)
                train_loss.append(loss)
            elif line.startswith('test acc/loss'):
                parts = line.strip().split()
                acc = float(parts[2])
                loss = float(parts[3])
                test_acc.append(acc)
                test_loss.append(loss)

    return train_acc, train_loss, test_acc, test_loss

def find_convergence_epoch(test_loss, improvement_threshold=1e-4, patience=10):

    if len(test_loss) < patience:
        return None

    for start in range(len(test_loss) - patience):
        stable = True
        for i in range(start+1, start+patience+1):
            diff = test_loss[i-1] - test_loss[i]
            if diff > improvement_threshold:
                stable = False
                break
        if stable:
            return start+1  # 1-based index
    return None

def plot_results(train_acc, train_loss, test_acc, test_loss, save_folder):
    epochs = range(1, len(train_acc)+1)

    convergence_epoch = find_convergence_epoch(test_loss, 
                                               improvement_threshold=1e-4, 
                                               patience=10)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_acc, label='Train Accuracy', 
             linestyle=':', color='blue', linewidth=1)
    plt.plot(epochs, test_acc, label='Test Accuracy', 
             linestyle=':', color='orange', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN Train/Test Accuracy vs. Epoch')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    if convergence_epoch is not None and convergence_epoch <= len(test_acc):
        conv_acc_value = test_acc[convergence_epoch - 1]
        plt.axvline(x=convergence_epoch, color='green', linestyle='--', linewidth=1)
        plt.text(convergence_epoch, conv_acc_value,
                 f'Convergence Epoch {convergence_epoch}\nAcc={conv_acc_value:.4f}',
                 color='green', fontsize=9, ha='left', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='green'))

    acc_plot_path = os.path.join(save_folder, 'cnn_training_accuracy_vs_epoch.png')
    plt.savefig(acc_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_loss, label='Train Loss', 
             linestyle=':', color='blue', linewidth=1)
    plt.plot(epochs, test_loss, label='Test Loss', 
             linestyle=':', color='orange', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Train/Test Loss vs. Epoch')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    if convergence_epoch is not None and convergence_epoch <= len(test_loss):
        conv_loss_value = test_loss[convergence_epoch - 1]
        plt.axvline(x=convergence_epoch, color='red', linestyle='--', linewidth=1)
        plt.text(convergence_epoch, conv_loss_value,
                 f'Convergence Epoch {convergence_epoch}\nLoss={conv_loss_value:.10f}',
                 color='red', fontsize=9, ha='left', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    loss_plot_path = os.path.join(save_folder, 'cnn_training_loss_vs_epoch.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return convergence_epoch

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/.. = root
    data_dir = os.path.join(root_dir, 'data')

    all_folders = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d != 'backup']

    if len(all_folders) == 0:
        print("No folders found in 'data' directory.")
        return

    print("\nFolders found in 'data':")
    for folder in all_folders:
        print(folder)

    ans = input("Process all the above folders? (Y/N): ").strip().lower()
    if ans != 'y':
        print("Exiting without processing.")
        return

    processed_info = []  # To store (folder, convergence_epoch, acc_at_conv, loss_at_conv)
    for folder in all_folders:
        selected_path = os.path.join(data_dir, folder)
        result_file = os.path.join(selected_path, 'cnn_training_results.dat')
        if not os.path.exists(result_file):
            print(f"cnn_training_results.dat not found in folder {folder}. Skipping...")
            continue

        train_acc, train_loss, test_acc, test_loss = parse_cnn_results(result_file)

        if len(train_acc) == 0 or len(test_acc) == 0:
            print(f"No training logs found in {folder}. Skipping...")
            continue

        convergence_epoch = plot_results(train_acc, train_loss, test_acc, test_loss, selected_path)
        print(f"Figures saved in: {selected_path}")

        if convergence_epoch is not None:
            acc_at_conv = test_acc[convergence_epoch - 1]
            loss_at_conv = test_loss[convergence_epoch - 1]
            processed_info.append((folder, convergence_epoch, acc_at_conv, loss_at_conv))
        else:
            processed_info.append((folder, None, None, None))

    summary_file = os.path.join(data_dir, 'convergence_summary.dat')

    header_format = "{:<20} | {:<18} | {:<22} | {:<22}\n"
    row_format    = "{:<20} | {:<18} | {:<22} | {:<22}\n"

    with open(summary_file, 'w') as f:
        f.write(header_format.format("Folder", "Convergence_Epoch", 
                                     "Test_Accuracy_at_Conv", "Test_Loss_at_Conv"))
        f.write("-"*20 + "-+-" + "-"*18 + "-+-" + "-"*22 + "-+-" + "-"*22 + "\n")

        for folder, ce, acc, loss in processed_info:
            if ce is None:
                f.write(row_format.format(folder, "N/A", "N/A", "N/A"))
            else:
                acc_str = f"{acc:.4g}"
                loss_str = f"{loss:.4g}"
                f.write(row_format.format(folder, str(ce), acc_str, loss_str))

    print(f"Convergence summary saved to {summary_file}")


if __name__ == '__main__':
    main()
