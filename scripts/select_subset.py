"""Select 150 real and 150 fake samples from protocol file."""
import os

PROTOCOL_PATH = "data/external/protocol.txt"
OUT_DIR = "data/external"
N_REAL = 150
N_FAKE = 150

def select_subset():
    real, fake = [], []
    with open(PROTOCOL_PATH) as f:
        for line in f:
            parts = line.strip().split()
            # typical ASVspoof protocol format:
            # speaker_id  filename  -  attack_id  label
            filename = parts[1]
            label = parts[-1]  # "bonafide" or "spoof"

            if label == "bonafide" and len(real) < N_REAL:
                real.append(filename)
            elif label == "spoof" and len(fake) < N_FAKE:
                fake.append(filename)

            if len(real) >= N_REAL and len(fake) >= N_FAKE:
                break
    return real, fake

if __name__ == "__main__":
    real, fake = select_subset()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "selected_real.txt"), "w") as f:
        f.write("\n".join(real))
    with open(os.path.join(OUT_DIR, "selected_fake.txt"), "w") as f:
        f.write("\n".join(fake))
    print(f"Selected {len(real)} real, {len(fake)} fake")