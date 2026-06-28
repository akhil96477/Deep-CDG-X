import h5py
import numpy as np

filepath = 'PPI_data/CPDB_multiomics.h5'
print(f"Opening {filepath}...")
with h5py.File(filepath, 'r') as f:
    print("\nKeys in HDF5 file:")
    print(list(f.keys()))
    
    for key in f.keys():
        try:
            shape = f[key].shape
            dtype = f[key].dtype
            print(f"Dataset: {key}, Shape: {shape}, Dtype: {dtype}")
            if key == 'feature_names':
                print("All feature names:")
                names = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in f[key][:]]
                print(names)
            elif key == 'features':
                print(f"Sample features (first 5 nodes, first 10 columns):")
                print(f[key][:5, :10])
            elif key == 'gene_names':
                print(f"Sample gene names (first 5):")
                print(f[key][:5])
        except AttributeError:
            print(f"Group/Other object: {key}")
