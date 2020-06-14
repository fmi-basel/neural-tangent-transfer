import matplotlib.pyplot as plt


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize= 8)
plt.rc('ytick', labelsize= 8)
plt.rc('axes', labelsize= 8)
plt.rcParams.update({'font.size': 8})
plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

plt.rcParams['xtick.major.pad'] = '2'
plt.rcParams['ytick.major.pad'] = '1.2'


gen_kwargs_rand_sparse_plot = dict(label = 'Rand.', color = '#998ec3', linestyle = ':',lw = 1.5)

gen_kwargs_ntt_student_plot = dict(label = 'NTT', color = '#f1a340', linestyle = '-',lw = 1.5)

gen_kwargs_ntt_teacher_plot = dict( label = 'NTT teacher',  color = '#878787', linestyle = '--', lw = 1.5)

gen_kwargs_he_scaled_plot = dict( label = 'Scaled Rand.',  color = '#2b8cbe', linestyle = '-.', lw = 1.5)


# gen_kwargs_lw_logit_snip = dict( label = 'lw Logit-SNIP',  color = '#dd1c77', linestyle = ':', lw = 1.5)

gen_kwargs_lw_snip_plot = dict( label = 'Layerwise-SNIP',  color = '#a1d99b', linestyle = '-.', lw = 1.5)


gen_kwargs_logit_snip = dict( label = 'Logit-SNIP',  color = '#dd1c77', linestyle = ':', lw = 1.5)

gen_kwargs_snip_plot = dict( label = 'SNIP',  color = '#a1d99b', linestyle = '-.', lw = 1.5)

gen_kwargs_dense_plot = dict( color = '#878787', linestyle = '--', lw = 1.5)


# new_gen_kwags_subplots_adjust = dict(left=.15, bottom=.15, right=.94, top=.85, wspace = 0.5, hspace = 0.5)



width = 3.4

# two_fig_size = dict(width = width, height = 1/2 * (width / 1.618 + 0.5)) 
two_fig_size = dict(width = width, height = 1/2 * (width / 1.618 + 0.6)) 

four_fig_size = dict(width =width, height =(width / 1.618 + 0.5)) 

six_fig_size = dict(width =width, height = 3/2 * (width / 1.618 + 0.5)) 




