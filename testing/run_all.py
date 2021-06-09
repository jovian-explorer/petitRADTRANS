import os
import glob

scripts = glob.glob('*.py')
print(scripts)

for script in scripts:
    if script != 'run_all.py':
        if script != 'testing_TiO.py':
            continue
        os.system('python '+script)
