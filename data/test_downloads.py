from huggingface_hub import hf_hub_download
import json
path = hf_hub_download(repo_id='willdepueoai/parameter-golf', filename='manifest.json', subfolder='datasets', repo_type='dataset')
m = json.load(open(path))
print('Datasets in HF manifest:')
for d in m.get('datasets', []):
    print(' ', d['name'])