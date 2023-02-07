import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import minimize
from ellipse import LsqEllipse
from numba import njit



class Parameters(object):
    
    def __init__(self, a=1., alpha=1., tau=1., phi=1., gamma1=1., gamma2=1., freq_r=None, omega=None, n=1):
        self.a, self.alpha, self.tau, self.phi, self.gamma1, self.gamma2,  self.freq_r, self.omega =\
        a, alpha, tau, phi, gamma1, gamma2, freq_r, omega
        
        if (freq_r == None): self.freq_r = np.ones(shape=n)
        if (omega == None): self.omega = np.ones(shape=n)
        
        self.error_abs = np.ones(shape=n)
        self.error_arg = np.ones(shape=n)
        self.error_ellipse = np.ones(shape=n)
        self.error_canonical_arg = np.ones(shape=n)
        self.error_canonical_real = np.ones(shape=n)
            
    def get_parameters(self):
        return self.a, self.alpha, self.tau, self.phi, self.gamma1, self.gamma2,  self.freq_r, self.omega     
    
    
class Initial_Guess(object):
    
    def __init__(self, a=None, alpha=None, tau=None, phi=None, gamma1=None, gamma2=None, freq_r=None, omega=None, n=1):
        self.a, self.alpha, self.tau, self.phi, self.gamma1, self.gamma2,  self.freq_r, self.omega =\
        a, alpha, tau, phi, gamma1, gamma2, freq_r, omega
        
        if (freq_r == None): self.freq_r = np.ones(shape=n)
        if (a == None): self.a = np.ones(shape=n)
        if (alpha == None): self.alpha = np.ones(shape=n)
        if (tau == None): self.tau = np.ones(shape=n)
        if (phi == None): self.phi = np.ones(shape=n)
        if (gamma1 == None): self.gamma1 = np.ones(shape=n)
        if (gamma2 == None): self.gamma2 = np.ones(shape=n)
        if (omega == None): self.omega = np.ones(shape=n)
        self.HW_abs_ids = np.zeros(shape=(n, 2))
        self.FW_abs_ids = np.zeros(shape=(n, 2))
        self.HWHM_abs = np.ones(shape=n)
        
        
    def find_HW_ids(self, y):
        HM = (max(y) + min(y))/2
        index_left, index_right = np.argmin(y), np.argmin(y)
        while (y[index_left] < HM and index_left > 0): index_left -= 1
        while (y[index_right] < HM and index_right < len(y) - 1): index_right += 1
        return np.array([index_left, index_right])
    
    def get_HW_abs_ids(self, y, cut=-1):
        ids = self.find_HW_ids(y)
        self.HW_abs_ids[cut] = ids
        return ids
    
    def get_FW_abs_ids(self, y, cut=-1):
        ids = self.find_HW_ids(y)
        delta = ids[1] - ids[0]
        ids = np.array([ids[0] - 3*delta//2, ids[1] + 3*delta//2])
        #ids = [ids[0] - delta, ids[1] + delta]
        if ids[0] < 0: ids[0] = 0
        if ids[1] >= len(y): ids[1] = len(y) - 1
        self.FW_abs_ids[cut] = ids
        return ids
    
    def get_HWHM_abs(self, x, y, cut=-1):
        idx_l, idx_r = self.get_HW_abs_ids(y, cut=cut)
        hwhm = (x[idx_r] - x[idx_l])/2
        self.HWHM_abs[cut] = hwhm
#         plt.scatter(x[idx_l], y[idx_l])
#         plt.scatter(x[idx_r], y[idx_r])
        return hwhm

    def get_parameters(self):
        return self.a, self.alpha, self.tau, self.phi, self.gamma1, self.gamma2,  self.freq_r, self.omega
    
    
    
class S21(object):

    def __init__(self, power, data, freq):
        self.power, self.data, self.freq = np.array(power), np.array(data), np.array(freq)
        
        n = len(power)
        
        self.real, self.imag = np.real(self.data), np.imag(self.data)
        self.abs = np.abs(self.data)
        self.arg = np.unwrap(np.angle(self.data))
        
        self.params = Parameters(n=n)
        self.ig = Initial_Guess(n=n)
        self.update_initial_guess()
        
    def update_data(self):
        self.real, self.imag = np.real(self.data), np.imag(self.data)
        self.abs = np.abs(self.data)
        self.arg = np.unwrap(np.angle(self.data))
        
        
    def update_initial_guess(self):
        for cut in range(len(self.power)):
            x, y = self.freq, self.abs[cut]
            _ = self.ig.get_HW_abs_ids(y, cut=cut)
            _ = self.ig.get_FW_abs_ids(y, cut=cut)
            _ = self.ig.get_HWHM_abs(x, y, cut=cut)
        
        
    def fit_abs(self, cut=-1, is_plot=False, e_info=False):
        x, y = self.freq, self.abs[cut]
        if is_plot: plt.plot(x, y)

        def  Lorentzian(x, x_0, alpha, gamma, y_0):
            return y_0 - (alpha/(gamma*np.pi)) * (1/(((x - x_0)/gamma)**2 + 1))
        hwhm = self.ig.get_HWHM_abs(x, y, cut=cut)
        p0 = [x[np.argmin(y)], (max(y) - min(y))*np.pi*hwhm, hwhm, max(y)]
        popt, pcov = curve_fit(Lorentzian, x, y, p0 = p0, maxfev=100000)
        #if is_plot: plt.plot(x, Lorentzian(x, *popt), label='Lorentzian')
        
        def Abs(freq, freq_r, phi, gamma1, gamma2, a, R):
            t = (freq_r - freq)/gamma2
            #r = ((gamma1/(2*gamma2))*1/((1 + t**2 + rabi_freq**2/(gamma1*gamma2))))
            r = ((gamma1/(2*gamma2))*1/((1 + t**2 + R)))
            return a*np.sqrt(1 - 2*r*(np.cos(phi) - t*np.sin(phi)) + (r**2)*(1 + t**2))
        freq_r, alpha, gamma, a = popt[0], popt[1], popt[2], popt[3]
        ids = np.arange(0, len(self.freq))
        #ids = np.arange(0, self.ig.get_FW_abs_ids(y)[1])
        ids = np.arange(*self.ig.get_FW_abs_ids(y))
        #if is_plot: plt.plot(x[ids], y[ids])
        p0 = [freq_r, 0., gamma, 2*gamma, a, 1.]
        popt, pcov = curve_fit(Abs, x, y, p0=p0, maxfev=100000)
        error = np.sqrt(np.sum((y - Abs(x, *popt))**2))
        if is_plot: plt.plot(x, Abs(x, *popt), label='S_21')
        
        #if is_plot: plt.legend()
        if e_info: return popt, pcov, error
        else: return popt, pcov
    
    def fit_arg(self, cut=-1, is_plot=False, e_info=False):
        x, y = self.freq, self.arg[cut]
        if is_plot: plt.plot(x, y)
        
        def linear(x, a, b): return a*x + b
        popt, pcov = curve_fit(linear, x, y)
        if is_plot: plt.plot(x, linear(x, *popt), label='S_21')
            
        def Arg(freq, freq_r, phi, gamma1, gamma2, alpha, rabi_freq, tau):
            t = (freq_r - freq) / gamma2
            r = ((gamma1 / (2 * gamma2)) * 1 / ((1 + t ** 2 + rabi_freq ** 2 / (gamma1 * gamma2))))
            return (alpha + freq*tau + 
                    np.arctan(-(t*np.cos(phi) + np.sin(phi))/(1/r + t*np.sin(phi) - np.cos(phi))))
        tau, alpha = popt[0], popt[1]
        hwhm = self.ig.get_HWHM_abs(self.freq, self.abs[cut], cut=cut)
        freq_r, phi, gamma1, gamma2, rabi_freq = x.mean(), 0., hwhm, 2*hwhm, 1.
        p0 = [freq_r, phi, gamma1, gamma2, alpha, rabi_freq, tau]
        popt, pcov = curve_fit(Arg, x, y, p0=p0, maxfev=100000)
        error = np.sqrt(np.sum((y - Arg(x, *popt))**2))
        if is_plot: plt.plot(x, Arg(y, *popt), label='S_21')
        if is_plot: plt.scatter(freq_r, Arg(freq_r, *popt))
        
        if e_info: return popt, pcov, error
        else: return popt, pcov
    
    def fit_canonical_arg(self, cut=-1, is_plot=False, e_info=False):
        x, y = self.freq, self.arg[cut]
        if is_plot: plt.plot(x, y)
        
        def arg(freq, freq_r, phi, gamma2): return phi + np.arctan((freq_r - freq)/gamma2)
        #ids = np.arange(len(self.freq))
        ids = np.arange(*self.ig.FW_abs_ids[cut], dtype=int)
        if is_plot: plt.plot(x[ids], y[ids])
        p0 = [self.params.freq_r[cut], 0., self.ig.HWHM_abs[cut]]
        popt, pcov = curve_fit(arg, x[ids], y[ids], p0=p0)
        error = np.sqrt(np.sum((y - arg(x, *popt))**2))
        if is_plot: plt.plot(x, arg(x, *popt))
        
        if e_info: return popt, pcov, error
        else: return popt, pcov
    
    
    def fit_canonical_real(self, cut=-1, is_plot=False, e_info=False):
        def lorentzian(x, freq_r, gamma, R): 
            return gamma/(1 + R + ((freq_r - x)/self.ig.gamma2[cut])**2)
        x, y = self.freq, self.real[cut]
        #t = (self.ig.freq_r[cut] - x)/self.ig.gamma2[cut]
        
        ids = np.arange(*self.ig.FW_abs_ids[cut], dtype=int)
        freq_r0 = self.ig.freq_r[cut]
        R0 = self.ig.HWHM_abs[cut]/self.ig.gamma2[cut]
        gamma0 = max(y)*(1 + R0)
        p0 = [freq_r0, gamma0, R0]
        popt, pcov = curve_fit(lorentzian, x[ids], y[ids], p0=p0)
        error = np.sqrt(np.sum((y - lorentzian(x, *popt))**2))
        if is_plot:
            plt.plot(x, y)
            plt.plot(x[ids], y[ids])
            plt.plot(x, lorentzian(x, *popt))
        
        if e_info: return popt, pcov, error
        else: return popt, pcov
    
    
    def fit_ellipse(self, cut=-1, is_plot=False, e_info=False):
        @njit
        def ellipse(tau, Ax, Ay, phi, a):
            x1, y1 = Ax*(1 + np.cos(tau)), Ay*np.sin(tau)
            z1 = (x1 + 1j*y1)
            z2 = a*(1 - z1*np.exp(1j*phi))
            x, y = np.real(z2), np.imag(z2)
            return x, y

        def error(p, x, y):
            Ax, Ay, phi, a = p
            x0, y0 = a*(1 - Ax*np.cos(phi)), -a*Ax*np.sin(phi)
            err = 0
            for i in range(0, len(x)):
                alpha = np.arctan2((y[i] - y0), (x[i] - x0))
                tau = np.arctan((Ax/Ay)*np.tan(alpha - phi))
                tmp_x, tmp_y = ellipse(tau, Ax, Ay, phi, a)
                alpha_alt = np.arctan2((tmp_y - y0), (tmp_x - x0))
                if np.sign(alpha) != np.sign(alpha_alt): 
                    tau += np.pi
                    tmp_x, tmp_y = ellipse(tau, Ax, Ay, phi, a)
                err += ((tmp_x - x[i])**2 + (tmp_y - y[i])**2)
            return err
        
        def get_ellipse(p, x, y):
            Ax, Ay, phi, a = p
            x0, y0 = a*(1 - Ax*np.cos(phi)), -a*Ax*np.sin(phi)
            ell = []
            for i in range(0, len(x)):
                alpha = np.arctan2((y[i] - y0), (x[i] - x0))
                tau = np.arctan((Ax/Ay)*np.tan(alpha - phi))
                tmp_x, tmp_y = ellipse(tau, Ax, Ay, phi, a)
                alpha_alt = np.arctan2((tmp_y - y0), (tmp_x - x0))
                if np.sign(alpha) != np.sign(alpha_alt):
                    tau += np.pi
                    tmp_x, tmp_y = ellipse(tau, Ax, Ay, phi, a)
                ell.append(tmp_x + 1j*tmp_y)
            return ell


        def fit(x, y):
            data = np.array(list(zip(x, y)))
            reg = LsqEllipse().fit(data)
            center, width, height, phi = reg.as_parameters()

            a0 = abs(center[0] + np.sqrt(height**2 - center[1]**2))
            #a0 = self.ig.a[cut]
            phi0 = min(abs(phi), abs(np.pi - phi))
            Ax0, Ay0 = abs(width/a0), abs(height/a0)
            p0 = [Ax0, Ay0, phi0, a0]

            res = minimize(error, x0=p0, args=(x, y))
            popt = res.x

            return popt
        
        ids = np.arange(*self.ig.FW_abs_ids[cut], dtype=int)
        x, y = self.real[cut], self.imag[cut]
        popt = fit(x[ids], y[ids])
        baseline_deleted_ellipse = get_ellipse(popt, x, y)
        
        if is_plot:
            plt.axis('equal')
            plt.scatter(x, y, 1)
            plt.scatter(x[ids], y[ids], 1)
            plt.scatter(np.real(baseline_deleted_ellipse), np.imag(baseline_deleted_ellipse), c='r', s=1)
            plt.show()
        
        if e_info: return popt, baseline_deleted_ellipse, error(popt, x[ids], y[ids])
        else: return popt, baseline_deleted_ellipse
            

    def remove_background(self, remove_baseline=False, is_plot=False):
        for cut in range(len(self.power)):
#             try:
                popt, pcov, error = self.fit_abs(cut=cut, e_info=True)
                freq_r, a = popt[0], popt[4]
                self.params.freq_r[cut] = freq_r
                self.params.error_abs[cut] = error
                
                popt, pcov, error = self.fit_arg(cut=cut, e_info=True)
                alpha, tau = popt[-3], popt[-1]
                self.params.error_arg[cut] = error
                self.ig.alpha[cut] = alpha
                self.ig.tau[cut] = tau
                
                self.data[cut] /= np.exp(1j*alpha)*np.exp(1j*tau*self.freq)
                self.update_data()

                popt, baseline_deleted_ellipse, error = self.fit_ellipse(cut=cut, is_plot=is_plot, e_info=True)
                if is_plot: plt.show()
                a = popt[3]
                self.params.error_ellipse[cut] = error
                self.ig.a[cut] = a

                if remove_baseline == True: self.data[cut] = baseline_deleted_ellipse
                self.data[cut] /= a

                self.data[cut] = 1 - self.data[cut]
                self.update_data()
                
#             except: self.data[cut] = np.empty(shape=len(data[cut]))
        self.update_data()
                
    
    def remove_mismatch(self, is_plot=False):
        for cut in range(len(self.power)):
            try:
                popt, pcov, error = self.fit_canonical_arg(cut=cut, is_plot=is_plot, e_info=True)
                if is_plot: plt.show()
                freq_r, phi, gamma2 = popt[0], popt[1], popt[2]
                self.params.error_canonical_arg[cut] = error
                self.ig.freq_r[cut] = freq_r
                self.ig.phi[cut] = phi
                self.ig.gamma2[cut] = gamma2
                
                
                self.data[cut] /= np.exp(1j*phi)
                
            except: self.data[cut] = np.empty(shape=len(data[cut]))
        self.update_data()
        
        
    def fit_lorentzian(self, is_plot=False):
        for cut in range(len(self.power)):
            try:
                popt, pcov, error = self.fit_canonical_real(cut=cut, is_plot=is_plot, e_info=True)
                if is_plot: plt.show()
                gamma, R = popt[1], popt[2]
                self.params.error_canonical_real[cut] = error
                self.ig.gamma1[cut] = 2*gamma*self.ig.gamma2[cut]
                self.ig.omega[cut] = R*2*gamma*(self.ig.gamma2[cut])**2
            except: self.data[cut] = np.empty(shape=len(data[cut]))
        
    
    def get_parameters(self):
        return self.params.get_parameters()