import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

HOME = Path(__file__).parent
output_path = HOME.joinpath('./output')
img_path = HOME.joinpath('./img')
img_path.mkdir(parents=True, exist_ok=True)

key = 'd1' # same as 'd2' and 'd3'
#key = 'd4' # same as 'd5'

### Load data ###
fout_monitor = output_path.joinpath(f'{key}_monitor.pk')
with open(fout_monitor, 'rb') as file:
    monitor = pickle.load(file)
#print(monitor)

### Plotting ###
lab_sz = 17
tic_sz = lab_sz-1
leg_sz = lab_sz-1
font_fam = 'Times New Roman'
lin_wd = 2.5

#nfun_max = 125 # d4
nfun_max = 25 # d1
nrule_max = 125
nbr_max = 50000

plt.figure()
plt.plot(monitor['pf_up'], label='Upper bound', linewidth=lin_wd)
plt.plot(monitor['pf_low'], linestyle='--', label='Lower bound', linewidth=lin_wd)
plt.yscale('log')
plt.ylim(1e-2, 1)
plt.xlim(0,nfun_max)
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
plt.ylim(0, nbr_max)
plt.xlim(0,nfun_max)
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
plt.ylim(0, nrule_max)
plt.xlim(0,nfun_max)
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

print(sum( monitor['time'] ))
