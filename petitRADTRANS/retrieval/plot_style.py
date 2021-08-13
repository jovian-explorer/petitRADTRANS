"""
This file creates a default plotting style for all pRT plots

All of these can be changed when calling most plotting functions.
This will affect the matplotlib rcParams, which can be reset to
the default values after pRT is finished.
"""

import matplotlib as mpl
import os
have_display = bool(os.environ.get('DISPLAY', None))
if not have_display:
    mpl.use('Agg')
prt_colours = ['#009FB8','#FF695C', '#70FF92',  '#FFBB33', '#6171FF', "#FF1F69", "#52AC25", '#E574FF', "#FF261D", "#B429FF" ]
font = {'family' : 'serif',
        'size'   : 24}
lines = {'markeredgecolor' : 'k',
         'markersize' : 8}

xtick = {'top' : True,
         'bottom' : True,
         'direction' : 'in',
         'labelsize' : 19}

ytick = {'left' : True,
         'right' : True,
         'direction' : 'in',
         'labelsize' : 19}
xmin = {'size': 5,
        'visible' : True}
ymin = {'size': 5,
        'visible' : True}
xmaj = {'size': 10}
ymaj = {'size': 10}
axes = {'labelsize' : 26,
        'prop_cycle': mpl.cycler(color=prt_colours),
        'titlesize' : 32}
figure = {'titlesize' : 32,
          'figsize' : (16,10),
          'dpi': 300,
          'autolayout' : True}
legend = {'fancybox' : True,
          'fontsize' : 20}
scatter = {'marker' : 'o',
           'edgecolors' : 'k'}


print("Using pRT Plotting style!")
mpl.rc('font', **font)
mpl.rc('lines',**lines)
mpl.rc('xtick',**xtick)
mpl.rc('xtick.minor',**xmin)
mpl.rc('xtick.major',**xmaj)
mpl.rc('ytick',**ytick)
mpl.rc('ytick.minor',**ymin)
mpl.rc('ytick.major',**ymaj)
mpl.rc('axes',**axes)
mpl.rc('figure',**figure)
mpl.rc('legend',**legend)
mpl.rc('scatter',**scatter)