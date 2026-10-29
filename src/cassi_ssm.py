import sys

import numpy as np
from scipy.sparse import coo_array

from src.polars_io import read_whitespace


# Pyhthon script for converting CASSI output to BridGE's sparse interaction network format,
# the same .npz layout compute_interactions.py writes (read back with load_sparse_array).
# inputs:
#    plinkbimfile: plink .bim file of the data CASSI ran on (sets the number of SNPs)
#    cassifile: CASSI output file
#    gitype: genetic interaction type
#            'lin': logistic regression test
#            'lr': logistic regression test
#            'je': joint effects
#            'awu': adjusted Wu method
#            'afe': adjusted fast epistasis
#            'wz': Wellek Ziegler method
#    outputfile: output file name (.npz)

def save_sparse_array(path, arrays, compressed=False):
    out = {}
    for name, A in arrays.items():
        A = A.tocsr()
        out[f"{name}__data"] = A.data
        out[f"{name}__indices"] = A.indices
        out[f"{name}__indptr"] = A.indptr
        out[f"{name}__shape"] = np.array(A.shape)
    (np.savez_compressed if compressed else np.savez)(path, allow_pickle=False, **out)
    
def to_csr_network(rows, cols, values, p):
    """Symmetric p x p csr_array from (row, col, value) triples.

    A pair listed more than once keeps its last value, and each pair is mirrored
    with the elementwise max, as the earlier dense version did.
    """
    keys = rows * p + cols
    _, last_reversed = np.unique(keys[::-1], return_index=True)
    keep = keys.size - 1 - last_reversed
    network = coo_array((values[keep], (rows[keep], cols[keep])), shape=(p, p)).tocsr()
    network = network.maximum(network.T).tocsr()
    network.eliminate_zeros()
    return network

def main():
	if len(sys.argv) < 5:
		sys.exit('not enough parameters')
	plinkbimfile = sys.argv[1]
	cassifile = sys.argv[2]
	gittype = sys.argv[3]
	outputfile = sys.argv[4]

	# load CASSI interaction file
	ssm_cassi = read_whitespace(cassifile)

	# load bim file
	bim = read_whitespace(plinkbimfile, has_header=False)

	# number of SNPs
	p = bim.shape[0]

	x = ssm_cassi['SNP1'].to_numpy() - 1
	y = ssm_cassi['SNP2'].to_numpy() - 1
 
	def col(name):
		return ssm_cassi[name].to_numpy()

	# -log10 p-value column, and which interactions are protective / risk
	if gittype == 'lr':
		# logistic regression (cassi)
		z = -np.log10(col('LR_P'))
		protective = col('LR_LOG_OR') < 0
		risk = col('LR_LOG_OR') > 0
	elif gittype == 'lin':
		# Linear Regression
		z = -np.log10(col('LIN_P'))
		protective = col('LIN_BETA') < 0
		risk = col('LIN_BETA') >= 0
	elif gittype in ('je', 'awu', 'afe'):
		# joint effect (cassi), Adjusted Wu, Adjusted Fast Epistasis
		prefix = gittype.upper()
		z = -np.log10(col(f'{prefix}_CC_P'))
		case_log_or = col(f'{prefix}_CASE_LOG_OR')
		ctrl_log_or = col(f'{prefix}_CTRL_LOG_OR')
		protective = ctrl_log_or > case_log_or
		risk = ctrl_log_or <= case_log_or
	elif gittype == 'wz':
		# Wellek Ziegler
		z = -np.log10(col('WZ_CC_P'))
		protective = col('WZ_CTRL_R') > col('WZ_CASE_R')
		risk = col('WZ_CTRL_R') <= col('WZ_CASE_R')
	else:
		sys.exit(f"unknown GI type {gittype!r}; expected one of lr, lin, je, awu, afe, wz")


	# save to file
	save_sparse_array(outputfile, {
		'risk': to_csr_network(x[risk], y[risk], z[risk], p),
		'protective': to_csr_network(x[protective], y[protective], z[protective], p),
	})

if __name__ == '__main__':
    main()
    