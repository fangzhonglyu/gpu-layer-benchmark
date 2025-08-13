# Install Python Dpendencies
pip3 install pandas
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
pip3 install pynvml

# Enable persistent GPU mode
nvidia-smi -pm 1