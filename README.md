# Installation 
```
git clone git@github.com:jieunbyun/BNS-JT.git
cd <BNS-JT dir>

# using venv
python3 -m venv <venv dir>
source <venv dir>/bin/activate
pip install -r requirements_py3.9.txt

# using conda
conda env create --name bns --file BNS_JT_py3.9.yml
conda activate bns
```

# Tests
```
pytest -v -s
```

# Examples 
Please take a look at 'notebooks' directory, which is under continual update.
