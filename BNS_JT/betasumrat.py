import mpmath
import numpy as np
from scipy import stats, integrate

# Adapted Lee Yingtong Li's betarat_gen function (https://yingtongli.me/git/scipy-yli/)
# Ref: Pham-Gia T. Distributions of the ratios of independent beta variables and applications. Communications in Statistics: Theory and Methods. 2000;29(12):2693–715. doi: 10.1080/03610920008832632

class BetaSumRat(stats.rv_continuous):
	"""
	Ratio of X/(X+Y) where X=Beta(a1,b1) and Y=Beta(a1,b1) are s.i.
	"""
	
	def _do_vectorized(self, func, x, a1, b1, a2, b2):
		"""Helper function to call the implementation over potentially multiple values"""
		
		x = np.atleast_1d(x)
		result = np.zeros(x.size)
		for i, (x_, a1_, b1_, a2_, b2_) in enumerate(zip(x, np.pad(a1, x.size, 'edge'), np.pad(b1, x.size, 'edge'), np.pad(a2, x.size, 'edge'), np.pad(b2, x.size, 'edge'))):
			result[i] = func(x_, a1_, b1_, a2_, b2_)
		return result
	
	def _pdf_one(self, t, a1, b1, a2, b2):
		"""PDF for the distribution, given by Pham-Gia"""
		
		if t <= 0:
			return 0
		elif t <= 0.5:
			A = mpmath.beta(a1, b1) * mpmath.beta(a2, b2)
			term1 = 1 / A / (1-t)**2
			term2 = mpmath.power(t/(1-t), a1 - 1)
			term3 = mpmath.beta(a1+a2, b2) * mpmath.hyp2f1(a1 + a2, 1 - b1, a1 + a2 + b2, (t)/(1-t))
		elif t<=1:
			A = mpmath.beta(a1, b1) * mpmath.beta(a2, b2)
			term1 = 1 / (A * t**2)
			term2 = mpmath.power(1/t-1, a2 - 1)
			term3 = mpmath.beta(a1+a2, b1) * mpmath.hyp2f1(a1 + a2, 1 - b2, a1 + a2 + b1, 1/t-1)
		else:
			return 0
		
		return float(term1 * term2 * term3)
	
	def _pdf(self, w, a1, b1, a2, b2):
		return self._do_vectorized(self._pdf_one, w, a1, b1, a2, b2)
	
	def _cdf_one(self, w, a1, b1, a2, b2):
		"""CDF for the distribution, by numerical integration"""
		
		if w <= 0:
			return 0
		else:
			pdf = lambda x: self._pdf_one(x, a1, b1, a2, b2)			
			cprob, _ = integrate.quad(pdf, 0, w)
			return cprob
	
	def _cdf(self, w, a1, b1, a2, b2):
		return self._do_vectorized(self._cdf_one, w, a1, b1, a2, b2)
	
	def _moment_one(self, k, a1, b1, a2, b2):
		"""Moments of the distribution, by numerical integration"""
		x_k = lambda x: x**k * self._pdf_one(x, a1, b1, a2, b2)
		moment = integrate.quad(x_k, 0, 1)
		return moment
	
	def _moment(self, k, a1, b1, a2, b2):
		return self._do_vectorized(self._moment_one, k, a1, b1, a2, b2)
	
	def _mean(self, a1, b1, a2, b2):
		return self._moment_one(1, a1, b1, a2, b2)

	def _var(self, a1, b1, a2, b2):
		mean = self._mean(a1, b1, a2, b2)
		E_x2 = self._moment(2, a1, b1, a2, b2)
		var = E_x2 - mean**2
		return var