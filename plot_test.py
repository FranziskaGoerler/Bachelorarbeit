import numpy as np
from matplotlib import pyplot as plt

def main():
    plt.axis([-50,50,0,10000])
    plt.ion()
    plt.show()
    ax = plt.gca()

    x = np.arange(-50, 51)
    for pow in range(1,5):   # plot x^1, x^2, ..., x^4
        y = [Xi**pow for Xi in x]
        ax.plot(x, y)
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

if __name__ == '__main__':
    main()