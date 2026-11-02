import pandas as pd
import numpy as np
import os

def process_msr_trace(input_csv, output_csv, server_category):
    print(f"Loading raw MSR trace: {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}. Please download an MSR Cambridge trace.")
        return

    cols = ['timestamp', 'hostname', 'disk_num', 'io_type', 'offset', 'size', 'response_time']
    df = pd.read_csv(input_csv, names=cols)
    
    # Convert Windows Filetime to standard hours (0-23)
    df['hour'] = (df['timestamp'] / (10**7 * 3600)) % 24
    df['hour'] = df['hour'].astype(int)
    
    # Map raw block offsets to File IDs and convert sizes to MB
    df['file'] = 'block_' + df['offset'].astype(str)
    df['file_size_mb'] = np.ceil(df['size'] / (1024 * 1024)).astype(int)
    df['file_size_mb'] = df['file_size_mb'].replace(0, 1) 
    
    # Inject Semantic Metadata (Bridging the Semantic Gap)
    df['category'] = server_category
    df['user_priority'] = df['io_type'].apply(lambda x: 1.0 if str(x).strip().lower() == 'read' else 0.0)
    
    final_df = df[['file', 'hour', 'category', 'user_priority', 'file_size_mb']].head(50000)
    final_df.to_csv(output_csv, index=False)
    print(f"Successfully formatted trace! Saved to {output_csv}")

if __name__ == "__main__":
    # Ensure you have downloaded a trace like 'src1.csv' from the SNIA repository
    process_msr_trace('src1.csv', 'access_trace.csv', server_category='Development')