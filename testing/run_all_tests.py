import os
import glob

scripts = glob.glob('*.py')
print(scripts)

for script in scripts:
    if script != 'run_all.py':
        os.system('python '+script)
