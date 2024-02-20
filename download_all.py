import glob
import subprocess
import sys

# Check if the correct number of arguments are passed
if len(sys.argv) != 3:
    print("Usage: python run_scripts.py <from_date> <to_date>")
    sys.exit(1)

# Extract from_date and to_date from command-line arguments
from_date = sys.argv[1]
to_date = sys.argv[2]

# Directory containing the Python files
directory = "./download_scripts"  # Current directory; adjust as needed

# Iterate over each Python file in the directory
python_files = glob.glob(f"{directory}/*.py")
for file in python_files:
    # Construct the command to run
    command = f"python3 {file} --from_date {from_date} --to_date {to_date}"
    
    # Execute the command
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Print the output and error (if any)
    print("Output:", result.stdout.decode('utf-8'))
    if result.stderr:
        print("Error:", result.stderr.decode('utf-8'))

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Success: {command}")
        print(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e.stderr.decode('utf-8')}")

def gunzip_files(directory):
    # Construct the command to gunzip all .gz files in the directory
    command = f"gunzip {directory}/*.gz"
    
    # Execute the command
    print(f"Executing: {command}")
    run_command(command)

# Replace '/path/to/directory' with the actual directory containing the .gz files
directory = "./datasets"
# Call gunzip_files function after all downloads are done
gunzip_files(directory)
print("gunzip operation completed.")

print("All commands executed.")
