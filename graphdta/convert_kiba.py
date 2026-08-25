"""将 kiba 数据转换为 GraphDTA 格式"""
import json, pickle, random, numpy as np, pandas as pd
from pathlib import Path
from collections import OrderedDict
from rdkit.Chem import MolFromSmiles, MolToSmiles

def canonical_smiles(smiles):
    mol = MolFromSmiles(smiles)
    return MolToSmiles(mol, isomericSmiles=True) if mol else None

def read_sequence(seq_file):
    with open(seq_file, 'r') as f:
        content = f.read().strip()
    lines = [l.strip() for l in content.split('\n') if not l.startswith('>')]
    seq = ''.join(lines)
    seq = ''.join(c for c in seq if c in 'ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy')
    return seq.upper() if seq else None

def find_file(target_dir, prefix, suffix):
    f = target_dir / f"{target_dir.name}_{prefix}.{suffix}"
    if f.exists() and f.stat().st_size > 0:
        return f
    return None

def read_activities(af):
    df = pd.read_csv(af)
    if 'smiles' in df.columns and 'paffinity' in df.columns:
        return list(zip(df['smiles'].tolist(), df['paffinity'].values.astype(float)))
    if 'canonical_smiles' in df.columns and 'standard_value' in df.columns:
        return list(zip(df['canonical_smiles'].tolist(), df['standard_value'].values.astype(float)))
    raise ValueError(f"Unknown CSV format: {af}, columns: {list(df.columns)}")

def process_dataset(src_dir, dataset_name, output_base):
    src_dir = Path(src_dir)
    out_dir = Path(output_base) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Converting {dataset_name}: {src_dir} -> {out_dir}")
    print(f"{'='*60}")

    all_pairs = []
    n_ok = n_skip = 0
    for split in ['train_set', 'val_set', 'test_set']:
        sd = src_dir / split
        if not sd.exists():
            continue
        for cat in sorted(sd.iterdir()):
            if not cat.is_dir():
                continue
            for td in sorted(cat.iterdir()):
                if not td.is_dir():
                    continue
                af = find_file(td, 'processed_activities', 'csv')
                sf = find_file(td, 'processed_protein_sequence', 'txt')
                if not af or not sf:
                    n_skip += 1
                    continue
                seq = read_sequence(sf)
                if not seq:
                    n_skip += 1
                    continue
                try:
                    act_pairs = read_activities(af)
                except Exception as e:
                    print(f"  WARN: {af}: {e}")
                    n_skip += 1
                    continue
                n_ok += 1
                for smi, aff in act_pairs:
                    csmi = canonical_smiles(smi)
                    if csmi is None or not np.isfinite(aff):
                        continue
                    all_pairs.append((csmi, seq, float(aff), split))

    print(f"  Targets: {n_ok} ok, {n_skip} skipped; Pairs: {len(all_pairs)}")
    if not all_pairs:
        print("  No valid data!")
        return

    drug_set = OrderedDict()
    protein_set = OrderedDict()
    for smi, seq, aff, sp in all_pairs:
        if smi not in drug_set:
            drug_set[smi] = f"drug_{len(drug_set)}"
        if seq not in protein_set:
            protein_set[seq] = f"protein_{len(protein_set)}"

    # Kiba-specific: train_set and test_set contain the same proteins.
    # Split proteins into train/val/test (80/10/10) for cold-start evaluation.
    protein_ids = list(protein_set.values())
    random.shuffle(protein_ids)
    n_total = len(protein_ids)
    n_test = max(1, int(n_total * 0.15))
    n_val = max(1, int(n_total * 0.15))
    test_proteins = set(protein_ids[:n_test])
    val_proteins = set(protein_ids[n_test:n_test + n_val])
    target_split = {}
    sc = {'train_set': 0, 'val_set': 0, 'test_set': 0}
    for seq, pid in protein_set.items():
        if pid in test_proteins:
            s = 'test_set'
        elif pid in val_proteins:
            s = 'val_set'
        else:
            s = 'train_set'
        target_split[pid] = s
        sc[s] += 1

    print(f"  Proteins: {len(drug_set)} drugs x {len(protein_set)} proteins")
    print(f"  Split: train={sc['train_set']}, val={sc['val_set']}, test={sc['test_set']}")

    # Build Y matrix
    d2i = {s: i for i, s in enumerate(drug_set)}
    p2i = {s: i for i, s in enumerate(protein_set)}
    Y = np.full((len(drug_set), len(protein_set)), np.nan, dtype=np.float32)
    n_filled = n_avg = 0
    for smi, seq, aff, sp in all_pairs:
        i, j = d2i[smi], p2i[seq]
        if np.isnan(Y[i, j]):
            Y[i, j] = aff
            n_filled += 1
        else:
            Y[i, j] = (Y[i, j] + aff) / 2.0
            n_avg += 1

    print(f"  Y: {Y.shape}, {n_filled} filled, {n_avg} averaged, "
          f"non-NaN ratio: {np.sum(~np.isnan(Y)) / Y.size:.4f}")

    # Save
    with open(out_dir / 'ligands_can.txt', 'w') as f:
        json.dump({drug_set[s]: s for s in drug_set}, f)
    with open(out_dir / 'proteins.txt', 'w') as f:
        json.dump({protein_set[s]: s for s in protein_set}, f)
    with open(out_dir / 'Y', 'wb') as f:
        pickle.dump(Y, f)
    with open(out_dir / 'target_split.json', 'w') as f:
        json.dump(target_split, f, indent=2)

    # Per-split pair counts
    spc = {'train_set': 0, 'val_set': 0, 'test_set': 0}
    for j, seq in enumerate(protein_set):
        sp = target_split[protein_set[seq]]
        spc[sp] += int(np.sum(~np.isnan(Y[:, j])))
    print(f"  Pairs per split: {spc}")
    print(f"  [OK] {dataset_name} done!")

if __name__ == '__main__':
    import argparse

    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert KIBA to GraphDTA format")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=base / "data")
    args = parser.parse_args()
    process_dataset(args.input_dir, 'kiba', args.output_root)
