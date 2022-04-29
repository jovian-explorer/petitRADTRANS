import os
import glob

scripts = glob.glob('*.py')
print(scripts)

'''
for script in scripts:
    if script != 'run_all_tests.py':
        #if script != 'testing_TiO.py':
        #    continue
        os.system('python '+script)
'''

os.system('python test_intro.py')
os.system('python test_high_res.py')
os.system('python test_clouds.py')
os.system('python test_scatter_intro.py')
os.system('python test_analysis.py')
os.system('python test_chemistry.py')
os.system('python test_downsampling.py')
