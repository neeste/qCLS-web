import scipy.io as sio
import json
import numpy as np
import os

def export_catalog(mat_file, json_file):
    mat = sio.loadmat(mat_file)
    strmcpf = mat['strmcpf'] # (2000, 20, 10)
    strfar = mat['strfar']   # (2000, 1)

    listeners = []
    # Export the first 100 listeners
    for i in range(100):
        theta = np.zeros(210)
        for f in range(10):
            # mcpf for this freq
            theta[f*20:(f+1)*20] = strmcpf[i, :, f]
            # far for this freq
            theta[200 + f] = strfar[i, 0]
        
        listeners.append({
            "id": f"listener_{i}",
            "theta": theta.tolist()
        })

    with open(json_file, 'w') as f:
        json.dump(listeners, f)

    print(f"Exported 100 listeners from {os.path.basename(mat_file)} to {json_file}")

base_dir = '/Users/neely/Library/CloudStorage/OneDrive-FatherFlanagan\'sBoysHome/STN/cls/CLS2023/fit_tenfrq.2024'

export_catalog(f"{base_dir}/clspf_v2.mat", 'simulated_listeners_v2.json')
export_catalog(f"{base_dir}/clspf_v3.mat", 'simulated_listeners_v3.json')
