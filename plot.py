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

    for input in args.input:
        data = np.loadtxt(input, delimiter=' ')
        plt.plot(data[:, 0] - 1, data[:, 1], label=os.path.splitext(os.path.basename(input))[0])
    plt.xlabel("Rounds")
    plt.ylabel("Win Rate")
    plt.xlim(0, 120000)
    plt.ylim(0.0, 0.85)
    plt.legend()
    fname = os.path.splitext(os.path.basename(input))[0] + '.jpg'
    plt.savefig(fname)
    print("Figure saved as '%s'" % fname)
    plt.show()