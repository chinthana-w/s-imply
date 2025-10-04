import deepgate
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bench_to_embed(bench_path: str) -> tuple[torch.Tensor, torch.Tensor]:

    model = deepgate.Model()    # Create DeepGate
    model.load_pretrained()      # Load pretrained model
    model = model.to(device)
    
    print('[INFO] Parse Bench: ', bench_path)
    parser = deepgate.BenchParser()   # Create BenchParser

    graph = parser.read_bench(bench_path) # Parse Bench into Graph
    graph = graph.to(device) # type: ignore
    hs, hf = model(graph) # Model inference 

    return hs, hf

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