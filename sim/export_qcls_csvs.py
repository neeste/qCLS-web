import scipy.io as sio
import pandas as pd
import numpy as np

def generate_csv(cond):
    mat_path = f'/Users/neely/Library/CloudStorage/OneDrive-FatherFlanagan\'sBoysHome/STN/2024_Quick_Tests/qCLS_analysis/coef_qcls{cond}_audio_prior.mat'
    out_csv = f'mcpf_parameters_qcls{cond}_prior.csv'
    mat = sio.loadmat(mat_path)
    coeff = mat['coeff'] # (99, 20, 10)
    far = mat['far']     # (99, 10)
    frq = mat.get('frq', np.array([250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000])).flatten()
    
    rows = []
    num_listeners = coeff.shape[0]
    
    for i in range(num_listeners):
        for f_idx in range(10):
            freq_hz = frq[f_idx]
            false_alarm_rate = far[i, f_idx]
            coeffs = coeff[i, :, f_idx]
            
            # Enforce non-negative boundary intervals
            for k in range(11, 20):
                if coeffs[k] < 1e-4:
                    coeffs[k] = 1e-4
            
            bounds = np.cumsum(coeffs[10:20])
            row = [i+1, freq_hz, false_alarm_rate] + coeffs.tolist() + bounds.tolist()
            rows.append(row)
            
    cols = ['listener_id', 'freq_hz', 'false_alarm_rate'] + \
           [f'coeff_{str(k+1).zfill(2)}' for k in range(20)] + \
           [f'cu_{str(k).zfill(2)}' for k in range(5, 51, 5)]
           
    df = pd.DataFrame(rows, columns=cols)
    df = df[df['cu_50'] != 0]
    df.to_csv(out_csv, index=False)
    print(f"Exported {out_csv} with {df['listener_id'].nunique()} valid listeners.")

for c in [1, 5]:
    generate_csv(c)
