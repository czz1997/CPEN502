import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPEN 502 plotting tool')
    parser.add_argument('-input', type=str)
    args = parser.parse_args()

    data = np.loadtxt(args.input, delimiter=' ')

    plt.plot(data[:, 0] - 1, data[:, 1])
    plt.xlabel("Rounds")
    plt.ylabel("Win Rate")
    plt.xlim(0, data[:, 0].max() * 1.05)
    plt.ylim(0, data[:, 1].max() * 1.05)
    fname = os.path.splitext(os.path.basename(args.input))[0] + '.jpg'
    plt.savefig(fname)
    print("Figure saved as '%s'" % fname)
    plt.show()