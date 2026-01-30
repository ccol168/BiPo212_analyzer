import numpy as np

def CalculateConcentrationWithRadialCut (Cut,counts,error,efficiency,Total_duration) :

	Total_slice_volume = 4./3.*np.pi*Cut**3  # m^3

	print(f"Volume in {Cut} M = {Total_slice_volume} m^3")
	print("Total dataset duration =",Total_duration,"s")
	print("Efficiency =",efficiency)
	print("Events selected in the cut =",counts,"±",error)

	concentration = counts/efficiency/Total_duration/Total_slice_volume
	error_concentration = error/efficiency/Total_duration/Total_slice_volume
		
	Th_concentration = concentration / 4060 / 856 / 1e3
	error_Th_concentration = error_concentration / 4060 / 856 / 1e3

	print(f"Total detector rate in {Cut} m",counts/efficiency/Total_duration,"±",error/efficiency/Total_duration,"Bq")
	print(f"Estimated BiPo concentration in {Cut} m",concentration,"±",error_concentration,"Bq/m^3")
	print(f"Estimated Th-equivalent concentration in {Cut} m",Th_concentration,"±",error_Th_concentration,"g/g")

	return Th_concentration,error_Th_concentration

def CalculateConcentrationinFV (counts,error,efficiency,Total_duration) :

	Total_slice_volume = 4./3.*np.pi * 16.5**3 - 2*((1./3.) * np.pi * (1)**2 * (3*16.5-1))

	print(f"Volume in FV (R<16.5 m, |z|<15.5 m) = {Total_slice_volume} m^3")
	print("Total dataset duration =",Total_duration,"s")
	print("Efficiency =",efficiency)
	print("Events selected in the cut =",counts,"±",error)

	concentration = counts/efficiency/Total_duration/Total_slice_volume
	error_concentration = error/efficiency/Total_duration/Total_slice_volume
		
	Th_concentration = concentration / 4060 / 856 / 1e3
	error_Th_concentration = error_concentration / 4060 / 856 / 1e3

	print(f"Total detector rate in FV",counts/efficiency/Total_duration,"±",error/efficiency/Total_duration,"Bq")
	print(f"Estimated BiPo concentration in FV",concentration,"±",error_concentration,"Bq/m^3")
	print(f"Estimated Th-equivalent concentration in FV",Th_concentration,"±",error_Th_concentration,"g/g")

	return Th_concentration,error_Th_concentration

def CalculateConcentrationWithRadialCutinHalf (Cut,counts,error,efficiency,Total_duration) :

	Total_slice_volume = (4./3.*np.pi*Cut**3)/2. # m^3

	print("Total dataset duration =",Total_duration,"s")
	print("Efficiency =",efficiency)
	print("Events selected in the cut =",counts,"±",error)

	concentration = counts/efficiency/Total_duration/Total_slice_volume
	error_concentration = error/efficiency/Total_duration/Total_slice_volume
		
	Th_concentration = concentration / 4060 / 856 / 1e3
	error_Th_concentration = error_concentration / 4060 / 856 / 1e3

	print(f"Total detector rate in {Cut} m",counts/efficiency/Total_duration,"±",error/efficiency/Total_duration,"Bq")
	print(f"Estimated BiPo concentration in {Cut} m",concentration,"±",error_concentration,"Bq/m^3")
	print(f"Estimated Th-equivalent concentration in {Cut} m",Th_concentration,"±",error_Th_concentration,"g/g")
		
	return Th_concentration,error_Th_concentration
		

def Calculate212and214 (Cut,counts212,error212,efficiency212,counts214,error214,efficiency214,Total_duration) :

	Total_slice_volume = 4./3.*np.pi*Cut**3  # m^3

	print("Total dataset duration =",Total_duration,"s")

	concentration212 = counts212/efficiency212/Total_duration/Total_slice_volume
	error_concentration212 = error212/efficiency212/Total_duration/Total_slice_volume

	concentration214 = counts214/efficiency214/Total_duration/Total_slice_volume
	error_concentration214 = error214/efficiency214/Total_duration/Total_slice_volume

	print(f"Estimated BiPo 212 concentration in {Cut} m",concentration212,"±",error_concentration212,"Bq/m^3")
	print(f"Estimated Th-equivalent concentration in {Cut} m",concentration212/ 4060 / 856 / 1e3,"±",error_concentration212/ 4060 / 856 / 1e3,"g/g")

	print(f"Estimated BiPo 214 concentration in {Cut} m",concentration214,"±",error_concentration214,"Bq/m^3")
	print(f"Estimated U-equivalent concentration in {Cut} m",concentration214/ 12400 / 856 / 1e3,"±",error_concentration214/ 12400 / 856 / 1e3,"g/g")

	return concentration212,error_concentration212,concentration214,error_concentration214

def CalculateConcentration (Cut,counts,error,efficiency,Total_duration,Volume) :

	Total_slice_volume = Volume

	print(f"Volume in {Cut} M = {Total_slice_volume} m^3")
	print("Total dataset duration =",Total_duration,"s")
	print("Efficiency =",efficiency)
	print("Events selected in the cut =",counts,"±",error)

	concentration = counts/efficiency/Total_duration/Total_slice_volume
	error_concentration = error/efficiency/Total_duration/Total_slice_volume
		
	Th_concentration = concentration / 4060 / 856 / 1e3
	error_Th_concentration = error_concentration / 4060 / 856 / 1e3

	print(f"Total detector rate in {Cut} m",counts/efficiency/Total_duration,"±",error/efficiency/Total_duration,"Bq")
	print(f"Estimated BiPo concentration in {Cut} m",concentration,"±",error_concentration,"Bq/m^3")
	print(f"Estimated Th-equivalent concentration in {Cut} m",Th_concentration,"±",error_Th_concentration,"g/g")

	return Th_concentration,error_Th_concentration