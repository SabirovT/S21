import numpy as np
def make_baseline(freq=None, eps_baseline=0.1, n=2, delta=30.0e+6):
    n = 2
    
    if freq == None:
        freq = np.linspace(5.126e+09, 5.156e+09, 4501)
    
    baseline_periods = np.random.uniform(0.5*(2*np.pi/delta), (2*np.pi/delta), size=n)
    print(baseline_periods)
    
    baseline = np.array([eps_baseline*np.sin(baseline_periods[i]*freq) for i in range(n)])
    baseline = np.sum(baseline, axis=0)
    
    return baseline

def generate(freq=None, params=None, eps=0.1, eps_baseline=0., is_plot=False, baseline=[]):
    if params == None:
        params = Parameters(a=0.013, alpha=1.6e+3, tau=-3.3e-7, phi=-.05, gamma1=2.6e+6, gamma2=1.3e+6, 
                            freq_r=5.14e+9, omega=2.1e+6)
    
    a, alpha, tau, phi, gamma1, gamma2, freq_r, omega = params.get_parameters()
    
    if freq == None:
        freq = np.linspace(5.126e+09, 5.156e+09, 4501)
        #freq = np.linspace(5.106e+09, 5.176e+09, 4501)
        #freq = np.linspace(5.086e+09, 5.196e+09, 4501)
        
    noise = np.random.normal(0, eps, len(freq)) + 1j*np.random.normal(0, eps, len(freq))
    
    if len(baseline)==0: baseline = make_baseline(freq, eps_baseline=eps_baseline)
    
    t = (freq_r - freq)/gamma2
    r = ((gamma1/(2*gamma2))*1/((1 + t**2 + omega**2/(gamma1*gamma2))))
    data = [0., 0.]
    data[0] = a*np.exp(1j*alpha)*np.exp(1j*freq*tau)*(1 - r*(1 + 1j*t)*np.exp(-1j*phi))
    data[0] += noise
    data[0] *= 1. + baseline
    data[1] = a*np.exp(1j*alpha)*np.exp(1j*freq*tau)*(1 - r*(1 + 1j*t)*np.exp(-1j*phi))
    data[1] += noise
    data[1] *= 1. + baseline
    
    #data += (0. + baseline)
    
    if is_plot:
        plt.plot(freq, abs(data[0]))
        plt.plot(freq, a*(1. + baseline))
        print(np.std(baseline))
    
    return [0., 1.], data, freq
