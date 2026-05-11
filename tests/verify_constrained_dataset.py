import os
import random
import sys

sys.path.append(os.getcwd())

from src.ml.core.dataset import ReconvergentPathsDataset


def verify():
    print("Loading dataset...")
    # Use a small subset or the main dataset file if available
    dataset_path = "data/datasets/reconv_dataset.pkl"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        # Try to find any .pkl in data/datasets
        import glob

        pkls = glob.glob("data/datasets/*.pkl")
        if pkls:
            dataset_path = pkls[0]
            print(f"Using {dataset_path}")
        else:
            print("No dataset found.")
            sys.exit(1)

    ds = ReconvergentPathsDataset(
        dataset_path,
        inject_constraints=True,
        constraint_prob=1.0,  # Force constraints on all valid assignments
        add_logic_value=True,
        # small cache size to allow garbage collection
        cache_size=5,
    )

    print(f"Dataset size: {len(ds)}")

    constrained_count = 0
    samples_checked = 0

    # Check sample 1672 specifically (known failure case) + random others
    indices = [1672] + random.sample(range(len(ds)), min(3, len(ds)))

    print("\nChecking samples for injected constraints...")
    for idx in indices:
        try:
            # DEBUG: Inspect raw entry
            entry = ds.data[idx]
            info = entry["info"]
            file_path = entry["file"]

            print(f"\n--- Sample {idx} ---")
            if "pair_info" in info:
                pi = info["pair_info"]
                print(f"Structure: {pi.get('start')} -> {pi.get('reconv')}")
            else:
                print("No pair_info!")
                print(f"Info keys: {list(info.keys())}")
                print(f"Info content: {info}")

            sample = ds[idx]
            paths_emb = sample["paths_emb"]  # [P, L, D+3]
            node_ids = sample["node_ids"]

            # Logic values are at indices -3, -2, -1 in the last dimension
            # 0 -> [1, 0, 0] at indices [-3, -2, -1]
            # 1 -> [0, 1, 0]
            # X -> [0, 0, 1] (default initialization)

            has_cons_0 = (paths_emb[..., -3] > 0.9).any()
            has_cons_1 = (paths_emb[..., -2] > 0.9).any()

            if has_cons_0 or has_cons_1:
                constrained_count += 1
                print("Result: Constraints injected!")

                # Get constrained node IDs
                mask = (paths_emb[..., -3] > 0.9) | (paths_emb[..., -2] > 0.9)
                constrained_nodes = node_ids[mask].unique()
                print(f"Constrained nodes: {constrained_nodes.tolist()}")
            else:
                print("Result: NO constraints.")
                # If failed, try to debug why
                print("Investigation:")
                assignment = ds._solve_sample_assignment(info, file_path)
                if not assignment:
                    print("  -> Solver returned None (UNSAT/Timeout/Error)")
                else:
                    print(f"  -> Solver OK: Found {len(assignment)} assignments")
                    cons = ds._gen_constraints_for_sample(info, file_path)
                    print(f"  -> Generated constraints: {len(cons)}")
                    if cons:
                        print(f"  -> Constraints exist: {list(cons.keys())[:5]}...")
                        print("  -> Constraints in embedding? Checking node overlap...")
                        # Check if any constraint node is in node_ids
                        common = []
                        unique_nodes = node_ids.unique().tolist()
                        for cn in cons:
                            if cn in unique_nodes:
                                common.append(cn)
                        print(f"  -> Overlap with paths: {len(common)} nodes ({common[:5]}...)")
                        if common:
                            print("  -> ERROR: Overlap exists but embedding update failed?")
                        else:
                            print(
                                "  -> INFO: Constraints generated but none align with path nodes"
                                " (this is possible)"
                            )

            samples_checked += 1

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\nConstrained samples: {constrained_count}/{samples_checked}")

    if constrained_count > 0:
        print("SUCCESS: Constraints injected.")
        sys.exit(0)
    else:
        print("FAILURE: No constraints injected.")
        sys.exit(1)


if __name__ == "__main__":
    verify()
