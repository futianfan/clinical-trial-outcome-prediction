import numpy as np 
from scipy import stats

N=5

hint = np.array([0.772, 0.8, 0.813, 0.75, 0.74 ])
compose = np.array([0.689, .717, .723, .660, .653])

hint = np.array([0.868, 0.841, 0.823, 0.859, 0.845 ])
hint -=0.04
compose = np.array([.740, .797, .793, .770, .80])

# print(hint.mean(), hint.std())
# print(compose.mean(), compose.std()) 


def p_value(a, b):
	N = a.shape[0]
	var_a= a.var(ddof=1)
	var_b= b.var(ddof=1)
	s= np.sqrt((var_a+ var_b)/2)
	t= (a.mean()- b.mean())/(s*np.sqrt(2/N))
	df= 2*N- 2
	p= 1 - stats.t.cdf(t,df=df)
	return p 


mean_a = .726
mean_b = .715 
std_a = .015 
std_b = .016 


def pvalue_mean_std(mean_a, mean_b, std_a, std_b, N = 5):
	var_a = std_a**2 
	var_b = std_b**2 
	s= np.sqrt((var_a+ var_b)/2)
	t= (mean_a- mean_b)/(s*np.sqrt(2/N))
	df= 2*N- 2
	p= 1 - stats.t.cdf(t,df=df)
	return p 

	


print(pvalue_mean_std(mean_a, mean_b, std_a, std_b))

##  https://cloud.tencent.com/developer/article/1050641


