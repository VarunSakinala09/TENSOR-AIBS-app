import wfdb
import matplotlib.pyplot as plt

record = wfdb.rdrecord("dataset/raw/set-a/1002867")

print("Signal shape:", record.p_signal.shape)
print("Sampling frequency:", record.fs)
print("Number of leads:", record.n_sig)
print("Lead names:", record.sig_name)

wfdb.plot_wfdb(record=record, title="12-Lead ECG")
plt.show()