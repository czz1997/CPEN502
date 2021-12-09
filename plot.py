import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPEN 502 plotting tool')
    parser.add_argument('-input', type=str, nargs="+")
    args = parser.parse_args()

    max_y = -np.inf
    min_y = np.inf
    for input in args.input:
        data = np.loadtxt(input, delimiter=' ')
        plt.plot(data[:, 0] - 1, data[:, 1], label=os.path.splitext(os.path.basename(input))[0])
        max_y = max(max_y, data[:, 1].max())
        min_y = min(min_y, data[:, 1].min())
    plt.xlabel("Rounds")
    plt.ylabel("Win Rate")
    plt.xlim(0, 20000)
    plt.ylim(min_y * 0.95, min(max_y * 1.05, 1))
    plt.legend()
    plt.grid()
    fname = os.path.splitext(os.path.basename(input))[0] + '.jpg'
    plt.savefig(fname)
    print("Figure saved as '%s'" % fname)
    plt.show()