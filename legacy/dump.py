def estimate_period_zero_crossing(x, f_x):
    zero_crossings = np.where(np.diff(np.sign(f_x)))[0]
    if len(zero_crossings) > 1:
        T_est = np.mean(np.diff(x[zero_crossings])) * 2  # Assuming half-period per crossing
        return T_est
    else:
        return None

def estimate_period(x, f_x):
    peaks, _ = find_peaks(f_x)  # Find local maxima
    if len(peaks) > 1:
        T_est = np.mean(np.diff(x[peaks]))  # Average distance between peaks
        return T_est
    else:
        return None