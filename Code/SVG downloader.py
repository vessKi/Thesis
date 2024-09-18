import os

def check_ndjson_files(data_dir):
    all_files = os.listdir(data_dir)
    print(f"All files in data directory: {all_files}")

    for file in all_files:
        if file.endswith('.ndjson'):
            file_path = os.path.join(data_dir, file)
            print(f"Checking file: {file_path}")
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"File {file} has {len(lines)} lines.")
                    print(f"First few lines of {file}:")
                    for line in lines[:5]:  # Print first 5 lines for a preview
                        print(line.strip())
                else:
                    print(f"File {file} is empty.")

data_dir = r'C:\Users\kilia\Desktop\COOP\quickdraw-dataset\examples\nodejs'  # Directory containing .ndjson files
check_ndjson_files(data_dir)
