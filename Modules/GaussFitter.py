import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from iminuit import Minuit,cost

def extended_single_unbinned_iminuit_fit(data,cut_range) : 
	"""
	Unbinned log-likelihood fit using iMinuit with a single gaussian
	"""
    
	total_events = len(data)
	guess_signal = total_events*0.95
	guess_bkg = total_events*0.05

    
	def extended_single_gauss_pdf (x,s,b,mu,sigma):
		return s + b, s * norm.pdf(x, mu, sigma) + b * (1/(cut_range[1]-cut_range[0]))
	
	initial_guess = [
		guess_signal,  					# s
        guess_bkg,						# b
		1800,                       	# mu1
		60,                             # sigma
	]

	# Set reasonable bounds
	bounds = [
		(0, total_events),       # s
		(0, total_events),       # b
		(cut_range[0],cut_range[1]),     # mu1
		(0, None),		 # sigma
	]

	cost_func = cost.ExtendedUnbinnedNLL(data, extended_single_gauss_pdf)

	result = Minuit(cost_func,s=initial_guess[0],b=initial_guess[1],mu=initial_guess[2],sigma=initial_guess[3])
	result.limits["s"] = bounds[0]
	result.limits["b"] = bounds[1]
	result.limits["mu"] = bounds[2]
	result.limits["sigma"] = bounds[3]

	result.migrad()

	print(result)

	params_hat = result.values
	errors = result.errors
	return params_hat, errors

def fit_and_plot(data,cut_range,bins=200,savefig=False,fit_type="single",show_fig=True):
	"""
	Launcher for the fitting function + plotter
	Fit function used -> extended unbinned NLL fit made with iMinuit
	"""

	# ---------------- Define fit functions --------------------

	def single_gauss_pdf (x,s,b,mu,sigma):
		return s * norm.pdf(x, mu, sigma) + b * (1/(cut_range[1]-cut_range[0]))

	# ---------------- Cut data ----------------

	data = data[(data > cut_range[0]) & (data < cut_range[1])]


	# ===== EXTENDED SINGLE GAUSSIAN =====
	if fit_type == "single":
		ub_popt, ub_perr = extended_single_unbinned_iminuit_fit(data, cut_range)

		# ub_popt = [s, b, mu, sigma]
		s, b, mu, sigma = ub_popt
		s_err, b_err, mu_err, sigma_err = ub_perr

	else :
		print("Double gaussian fit not yet implemented\nThe only option currently available is \"single\" ")
		return None,None

	# ---------------- Make histogram of results for plotting ------------------

	counts, bin_edges = np.histogram(data, bins=bins, range=cut_range)
	bin_midpoints = (bin_edges[:-1] + bin_edges[1:])/2
	bin_size = (bin_edges[1] - bin_edges[0])
	
    # ---------------- Plot ----------------
	fig, (ax, ax_tbl) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [10, 4]})

	ax.errorbar(bin_midpoints,counts,yerr=np.sqrt(counts),marker="o",linestyle="",capsize=5,label="Data")
	ax.bar(bin_midpoints, counts, width=np.diff(bin_edges), align='center',alpha=0.2, color='gray', edgecolor='blue')

	x_fit = np.linspace(cut_range[0], cut_range[1], 1000)
	ax.plot(x_fit, single_gauss_pdf(x_fit, *ub_popt)*bin_size, "r-", label="Fit")

	ax.set_xlabel("NHits Po")
	ax.set_ylabel("Entries")
	ax.legend()

	# ---------------- Table ----------------
	def fmt(v, e):
		return f"{v:.3g} ± {e:.3g}"

	ax_tbl.axis("off")
	table_data = [["Best fit", ""]]

	
	table_data += [
		["212 Counts", fmt(s, s_err)],
		["Expected Bkg", fmt(b, b_err)],
		["μ", fmt(mu, mu_err)],
		["σ", fmt(sigma, sigma_err)],
	]

	tbl = ax_tbl.table(cellText=table_data, loc="center", cellLoc="center")
	tbl.scale(1.4, 1.6)

	plt.tight_layout()

	if savefig:
		plt.savefig(savefig)
	elif show_fig:
		plt.show()

	#return two dictionaries for better output readibility

	keys = ["signal","bkg","mu","sigma"]

	values = dict(zip(keys,ub_popt))
	errors = dict(zip(keys,ub_perr))

	return values, errors