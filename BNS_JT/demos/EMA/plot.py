import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

HOME = Path(__file__).parent
output_path = HOME.joinpath('./output')

key = 'ema1'
#key = 'ema2'

### Load data ###
fout_monitor = output_path.joinpath(f'monitor_{key}.pk')
with open(fout_monitor, 'rb') as file:
    monitor = pickle.load(file)
print(monitor)

### Plotting ###
img_path = HOME.joinpath('./img')
lab_sz = 17
tic_sz = lab_sz-1
leg_sz = lab_sz-1
font_fam = 'Times New Roman'
lin_wd = 2.5

plt.figure()
plt.plot(monitor['pf_up'], label='Upper bound', linewidth=lin_wd)
plt.plot(monitor['pf_low'], linestyle='--', label='Lower bound', linewidth=lin_wd)
#plt.yscale('log')
plt.ylim(0, 1)
plt.xlim(0,200)
plt.xlabel('No. of system function runs', fontsize=lab_sz, fontfamily=font_fam)
plt.ylabel('System failure probability', fontsize=lab_sz, fontfamily=font_fam)
plt.xticks(fontsize=tic_sz, fontfamily=font_fam)
plt.yticks(fontsize=tic_sz, fontfamily=font_fam)
if key[-1] == '1':
	plt.legend(prop={'family': font_fam, 'size': leg_sz})
plt.grid(True)
plt.savefig( img_path.joinpath(f'./pf_{key}.tiff'), format='tiff' )
plt.savefig( img_path.joinpath(f'./pf_{key}.png'), dpi=300 )
#plt.show()
plt.close()

plt.figure()
plt.plot(np.array(monitor['br_s_ns'])+np.array(monitor['br_f_ns'])+np.array(monitor['br_u_ns']), label='Total', linewidth=lin_wd)
plt.plot(monitor['br_s_ns'], linestyle='--', label='Survival', linewidth=lin_wd)
plt.plot(monitor['br_f_ns'], linestyle=':', label='Failure', linewidth=lin_wd)
plt.plot(monitor['br_u_ns'], linestyle='-.', label='Unknown', linewidth=lin_wd)
plt.yscale('log')
plt.ylim(0, 52000)
plt.xlim(0,200)
plt.xlabel('No. of system function runs', fontsize=lab_sz, fontfamily=font_fam)
plt.ylabel('No. of branches', fontsize=lab_sz, fontfamily=font_fam)
plt.xticks(fontsize=tic_sz, fontfamily=font_fam)
plt.yticks(fontsize=tic_sz, fontfamily=font_fam)
if key[-1] == '1':
	plt.legend(prop={'family': font_fam, 'size': leg_sz})
plt.grid(True)
plt.savefig( img_path.joinpath(f'./br_{key}.tiff'), format='tiff' )
plt.savefig( img_path.joinpath(f'./br_{key}.png'), dpi=300 )
#plt.show()
plt.close()

plt.figure()
plt.plot(np.array(monitor['r_s_ns'])+np.array(monitor['r_f_ns']), label='Total', linewidth=lin_wd)
plt.plot(monitor['r_s_ns'], linestyle='--', label='Survival', linewidth=lin_wd)
plt.plot(monitor['r_f_ns'], linestyle=':', label='Failure', linewidth=lin_wd)
#plt.yscale('log')
plt.ylim(0, 139)
plt.xlim(0,200)
plt.xlabel('No. of system function runs', fontsize=lab_sz, fontfamily=font_fam)
plt.ylabel('No. of non-dominated rules', fontsize=lab_sz, fontfamily=font_fam)
plt.xticks(fontsize=tic_sz, fontfamily=font_fam)
plt.yticks(fontsize=tic_sz, fontfamily=font_fam)
if key[-1] == '1':
	plt.legend(prop={'family': font_fam, 'size': leg_sz})
plt.grid(True)
plt.savefig( img_path.joinpath(f'./ru_{key}.tiff'), format='tiff' )
plt.savefig( img_path.joinpath(f'./ru_{key}.png'), dpi=300 )
#plt.show()
plt.close()