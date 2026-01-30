import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import expon,uniform
from tabulate import tabulate
from scipy.stats import chi2
import numdifftools as nd
from iminuit import Minuit,cost

def MakeExponentialFit(counts, bin_edges, fix_signal=None, fix_tau=None, fix_bkg=None):

    def expo_cdf (x,s,tau,b) :
          return s*expon.cdf(x,scale=tau) + b * uniform.cdf(x,loc=bin_edges[0],scale=bin_edges[-1])

    # Build free parameter list dynamically
    initial_guess = []
    fixed_parameters = []
    
    bounds = [
          (0,np.sum(counts)/0.315),   #s
          (0,None), #tau
          (0,np.sum(counts)) #b
	] 

    if fix_signal is None:
        initial_guess.append(0.9*np.sum(counts))
    else :
        initial_guess.append(fix_signal)
        fixed_parameters.append("s")
        
    if fix_tau is None:
        initial_guess.append(299./np.log(2))
    else :
        initial_guess.append(fix_tau)
        fixed_parameters.append("tau")
        
    if fix_bkg is None:
        initial_guess.append(0.1*np.sum(counts))
    else :
        initial_guess.append(fix_bkg)
        fixed_parameters.append("b")

    # Negative log-likelihood
    costfunction = cost.ExtendedBinnedNLL(counts,bin_edges,expo_cdf)

    # Run minimization
    m = Minuit(costfunction,s=initial_guess[0],tau=initial_guess[1],b=initial_guess[2])
    
    for element in fixed_parameters :
        m.fixed[element] = True
        
    for i,element in enumerate(["s","tau","b"]) :
        m.limits[element] = bounds[i]
    
    m.migrad()
    
    print(m)
    
    params_hat = m.values
    errors = m.errors
    
    return params_hat, errors


def ExponentialFit (data,cut_range,reBin = 1,fix_signal=None,fix_tau=None,fix_bkg=None,savefig=False) :
	"""
    Function that plots and calls the standard exponential fit, calculating fit chi squared
    The standard fit function is an extended binned NL fit
    """

	data = data[(data > cut_range[0]) & (data < cut_range[1])]

	#slice with the correct division
	total_counts, total_bin_edges = np.histogram(data,bins=200,range=[0,1200])
	total_bin_centers = (total_bin_edges[:-1] + total_bin_edges[1:]) / 2

	idx_min = np.searchsorted(total_bin_edges, cut_range[0], side="left")
	idx_max = np.searchsorted(total_bin_edges, cut_range[1], side="right") - 1

	# Slice the histogram to that range
	counts = total_counts[idx_min:idx_max]
	bin_edges = total_bin_edges[idx_min:idx_max + 1]
	bin_centers = total_bin_centers[idx_min:idx_max]

	# Optionally rebin
	if reBin > 1:
		# Group bins in chunks of size reBin
		n_rebin = len(counts) // reBin
		counts = counts[: n_rebin * reBin].reshape(-1, reBin).sum(axis=1)
		bin_edges = bin_edges[::reBin]
		bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

	params_hat, errors = MakeExponentialFit(counts, bin_edges, fix_signal, fix_tau, fix_bkg)

	signal = params_hat["s"]
	tau    = params_hat["tau"]
	bkg    = params_hat["b"]

	signal_err = errors["s"]
	tau_err    = errors["tau"]
	bkg_err    = errors["b"]
      
	halflife = tau*np.log(2)
	halflife_err = tau_err*np.log(2)

	# Bin centers
	bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

	# Compute expected counts per bin using the *cumulative function* (proper for binned likelihood)
	def expected_counts_per_bin(bin_edges, signal, tau, bkg):
		# CDF difference = expected counts in each bin
		return np.diff(
			signal * expon.cdf(bin_edges, scale=tau)
			+ bkg * uniform.cdf(bin_edges, loc=bin_edges[0], scale=bin_edges[-1])
		)

	expected = expected_counts_per_bin(bin_edges, signal, tau, bkg)

	# Compute likelihood-ratio chi2
	mask = (counts > 0) & (expected > 0)
	chi2_val = 2 * np.sum(expected[mask] - counts[mask] + counts[mask] * np.log(counts[mask] / expected[mask]))

	# Degrees of freedom (automatic: number of bins minus number of free parameters)
	n_free_params = sum(
		p is None
		for p in [fix_signal, fix_tau, fix_bkg]
	)
	ndof = len(counts) - n_free_params

	# p-value
	p_val = 1 - chi2.cdf(chi2_val, ndof)

	print(f"χ² = {chi2_val:.2f}, ndof = {ndof}, p-value = {p_val:.3f}")

	fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(10, 6),gridspec_kw={'width_ratios': [3, 1]})

	# Poisson errors = sqrt(N)
	y_errs = np.sqrt(counts)

	# Plot histogram with errors
	ax.errorbar(bin_centers, counts, yerr=y_errs, fmt='o', color='blue',
				markersize=4, capsize=2, label="Entries (with Poisson errors)")
	ax.bar(bin_centers, counts, width=np.diff(bin_edges), align='center',
       alpha=0.2, color='gray', edgecolor='blue')

	# Fitted curve
	x_fit = np.linspace(cut_range[0], cut_range[1], 500)
	y_fit = signal*expon.pdf(x_fit,scale=tau) + bkg * uniform.pdf(x_fit,loc=bin_edges[0],scale=bin_edges[-1])
	ax.plot(x_fit, y_fit*(bin_edges[1]-bin_edges[0]), 'r-', label='Best fit')

	ax.set_xlim(cut_range[0], cut_range[1])
	ax.set_ylim(bottom=0)
	ax.set_xlabel("Time delay (ns)")
	ax.set_ylabel("Events")
	ax.set_title("Time delays histogram and exponential fit")
	ax.legend()

	# --- Table ---
	ax_table.axis('off')

		# --- Prepare display strings ---
	param_names = ["Signal counts", "Half life", "Bkg counts", "Signal in ROI"]
	param_values = [signal, halflife, bkg, signal*0.315]
	param_errors = [signal_err, halflife_err, bkg_err, signal_err*0.315]
	param_fixed = [fix_signal is not None, fix_tau is not None, fix_bkg is not None]

		# Extend fixed flags for derived param
	param_fixed = [fix_signal is not None, fix_tau is not None, fix_bkg is not None, False]

	# Build table_data
	table_data = [["Best fit values", ""]]
	for name, val, err, is_fixed in zip(param_names, param_values, param_errors, param_fixed):
		if is_fixed:
			display_str = f"{val:.2f}" if name != "Half life" else f"{val:.1f}"
		else:
			display_str = f"{val:.2f} ± {err:.2f}" if name != "Half life" else f"{val:.1f} ± {err:.1f}"
		table_data.append([name, display_str])

	# Blank row
	table_data.append(["", ""])

	# Fit statistics
	table_data.append(["Fit statistics", ""])
	table_data.append([r"$\chi^2$", f"{chi2_val:.2f}"])
	table_data.append(["dof", f"{ndof}"])
	table_data.append(["p-value", f"{p_val:.3f}"])

	# --- Create table ---
	tbl = ax_table.table(cellText=table_data, loc="center", cellLoc="center")
	tbl.auto_set_font_size(False)
	tbl.set_fontsize(12)
	tbl.scale(1.5, 1.8)

	# Header row formatting
	tbl[(0, 0)].set_text_props(ha='center', weight='bold')
	tbl[(0, 1)].set_text_props(ha='center', weight='bold')
	tbl[(0, 0)].set_facecolor("#dddddd")

	# Highlight fixed parameters
	for i, is_fixed in enumerate(param_fixed):
		if is_fixed:
			tbl[(i + 1, 1)].set_facecolor("#cccccc")  # light grey

	# Fit statistics row
	fit_stat_row = len(param_names) + 2  # header + all param rows + blank row
	tbl[(fit_stat_row, 0)].set_text_props(ha='center', weight='bold')
	tbl[(fit_stat_row, 0)].set_facecolor("#dddddd")
	tbl[(fit_stat_row, 1)].set_text_props(ha='center', weight='bold')
      
	#blank divider rows
	tbl[(5, 1)].set_visible(False)

	# Fit statistics header second column (row 6, col 1)
	tbl[(6, 1)].set_visible(False)
	tbl[(5, 0)].set_visible(False)

	plt.tight_layout()
		
	if savefig :
		plt.savefig(savefig)
	else : plt.show()

	return 