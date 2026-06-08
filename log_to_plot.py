import json
import matplotlib.pyplot as plt

def plot_training_logs(log_path):
    epochs_data = {}

    # 1. Citirea și parsarea fișierului
    print(f"Citesc datele din: {log_path}")
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue 
                
                epoch = entry.get("epoch")
                if epoch is None:
                    continue
                    
                if epoch not in epochs_data:
                    epochs_data[epoch] = {}
                    
                if "train_loss" in entry:
                    epochs_data[epoch]["train_loss"] = entry["train_loss"]
                if "test_micro" in entry:
                    epochs_data[epoch]["test_micro"] = entry["test_micro"]
                if "test_macro" in entry:
                    epochs_data[epoch]["test_macro"] = entry["test_macro"]
                    
    except FileNotFoundError:
        print(f"Eroare: Nu am găsit fișierul {log_path}. Verifică calea.")
        return

    # 2. Pregătirea datelor
    sorted_epochs = sorted(epochs_data.keys())
    epochs, train_loss, test_micro, test_macro = [], [], [], []

    for ep in sorted_epochs:
        data = epochs_data[ep]
        if "train_loss" in data and "test_micro" in data:
            epochs.append(ep)
            train_loss.append(data["train_loss"])
            test_micro.append(data["test_micro"])
            test_macro.append(data["test_macro"])

    if not epochs:
        print("Nu s-au găsit date valide pentru plotare.")
        return

    # --- NOU: Găsirea celor mai bune scoruri ---
    best_micro_val = max(test_micro)
    best_micro_ep = epochs[test_micro.index(best_micro_val)]
    
    best_macro_val = max(test_macro)
    best_macro_ep = epochs[test_macro.index(best_macro_val)]

    # 3. Crearea graficului
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Evoluția Antrenării: UCSD Ped2', fontsize=16, fontweight='bold')

    # Sub-graficul 1: Train Loss
    ax1.plot(epochs, train_loss, color='tab:blue', linewidth=2, marker='o', markersize=4, label='Train Loss')
    ax1.set_ylabel('Eroare (Loss)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # Sub-graficul 2: Scorurile AUC
    ax2.plot(epochs, test_micro, color='tab:green', linewidth=2, marker='s', markersize=4, label='Micro AUC', alpha=0.7)
    ax2.plot(epochs, test_macro, color='tab:red', linewidth=2, marker='^', markersize=4, label='Macro AUC', alpha=0.7)
    
    # --- NOU: Marcarea celor mai bune scoruri pe grafic ---
    # Highlight Micro
    ax2.plot(best_micro_ep, best_micro_val, marker='o', color='darkgreen', markersize=10, markeredgecolor='black')
    ax2.annotate(f'Best Micro: {best_micro_val:.3f}\n(Ep {best_micro_ep})', 
                 xy=(best_micro_ep, best_micro_val), 
                 xytext=(15, -20), textcoords='offset points',
                 color='darkgreen', fontweight='bold', fontsize=10,
                 arrowprops=dict(arrowstyle="->", color='darkgreen'))

    # Highlight Macro
    ax2.plot(best_macro_ep, best_macro_val, marker='o', color='darkred', markersize=10, markeredgecolor='black')
    ax2.annotate(f'Best Macro: {best_macro_val:.3f}\n(Ep {best_macro_ep})', 
                 xy=(best_macro_ep, best_macro_val), 
                 xytext=(15, 10), textcoords='offset points',
                 color='darkred', fontweight='bold', fontsize=10,
                 arrowprops=dict(arrowstyle="->", color='darkred'))

    ax2.set_xlabel('Epoca', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Scor AUC', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    
    save_path = log_path.replace("log.txt", "learning_curve_annotated.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nGraficul a fost salvat cu succes ca: {save_path}")
    print(f"Cel mai bun Micro AUC: {best_micro_val:.4f} (Epoca {best_micro_ep})")
    print(f"Cel mai bun Macro AUC: {best_macro_val:.4f} (Epoca {best_macro_ep})")
    
    plt.show()

cale_fisier_log = "./Experiments/RUN06_fail/Checkpoints/log.txt"

plot_training_logs(cale_fisier_log)