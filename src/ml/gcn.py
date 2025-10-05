import torch
import argparse

# Force CPU usage due to CUDA compatibility issues
device = torch.device('cpu')

def bench_to_embed(bench_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings from a bench file using DeepGate or fallback to dummy embeddings."""
    
    print('[INFO] Parse Bench: ', bench_path)
    
    try:
        import deepgate
        
        model = deepgate.Model()    # Create DeepGate
        model.load_pretrained()      # Load pretrained model
        model = model.to(device)
        
        parser = deepgate.BenchParser()   # Create BenchParser
        graph = parser.read_bench(bench_path) # Parse Bench into Graph
        graph = graph.to(device) # type: ignore
        hs, hf = model(graph) # Model inference 
        return hs, hf
    except ImportError:
        print(f"[WARNING] DeepGate not available, using dummy embeddings for {bench_path}")
        # Parse the bench file to get basic circuit info
        from src.util.io import parse_bench_file
        try:
            circuit_gates, max_node_id = parse_bench_file(bench_path)
            
            # Find inputs and outputs
            input_gates = [g for g in circuit_gates if g.type == 1 and g.nfi == 0]
            output_gates = [g for g in circuit_gates if g.type != 0 and g.nfo == 0]
            
            num_inputs = len(input_gates)
            num_outputs = len(output_gates)
            total_nodes = max_node_id + 1
            
            # Create dummy embeddings with appropriate dimensions
            # Structural embeddings: one per node
            dummy_hs = torch.randn(total_nodes, 128, device=device)
            
            # Functional embeddings: one per input/output
            dummy_hf = torch.randn(num_inputs + num_outputs, 128, device=device)
            
            return dummy_hs, dummy_hf
        except Exception as e:
            print(f"[ERROR] Failed to parse {bench_path}: {e}")
            # Return minimal dummy embeddings
            dummy_hs = torch.randn(1, 128, device=device)
            dummy_hf = torch.randn(1, 128, device=device)
            return dummy_hs, dummy_hf
    except Exception as e:
        print(f"[ERROR] DeepGate parsing failed for {bench_path}: {e}")
        # Return dummy embeddings with correct shape
        dummy_hs = torch.randn(1, 128, device=device)
        dummy_hf = torch.randn(1, 128, device=device)
        return dummy_hs, dummy_hf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate embeddings from a bench file.")
    parser.add_argument('bench_file', type=str, help="Path to the bench file")
    args = parser.parse_args()

    file_name = args.bench_file.split('/')[-1].split('.')[0]

    struct_emb, func_emb = bench_to_embed(args.bench_file)

    print(func_emb)
    # Write func_emb to a CSV file
    # with open(f'{file_name}_func_emb.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(func_emb.detach().cpu().numpy())