import re
from collections import Counter

with open(r'd:/Diffusion_@ - Copy/Diffusion/paper_submission/dscsp_manuscript.tex','r',encoding='utf-8') as f:
    txt = f.read()

envs_begin = re.findall(r'\\begin\{([^}]+)\}', txt)
envs_end = re.findall(r'\\end\{([^}]+)\}', txt)
print('begin count', len(envs_begin), 'end count', len(envs_end))
cb = Counter(envs_begin); ce = Counter(envs_end)
for k in set(list(cb) + list(ce)):
    if cb[k] != ce[k]:
        print('MISMATCH', k, cb[k], ce[k])
print('Env types:', sorted(set(envs_begin)))

# brace balance
opens = txt.count('{')
closes = txt.count('}')
print('Braces: {', opens, '} ', closes)
