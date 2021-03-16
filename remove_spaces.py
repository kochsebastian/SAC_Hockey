from shutil import copyfile
import os
plots_names_spaces = os.listdir('plots/')
plots_names_spaces = [name for name in plots_names_spaces if '.py' not in name]

plots_names = [name.replace(' ','') for name in plots_names_spaces]
for name_space,name in zip(plots_names_spaces,plots_names):
    copyfile('plots/'+name_space, 'formatted_plots/'+name)