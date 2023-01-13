from colorsys import hls_to_rgb
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib import colors


def get_red_map(n):
    cdict = {'red':   ((0.0,  0.3, 0.3),
                       (1.0,  1.0, 1.0)),

             'green': ((0.0,  0.1, 0.1),
                       (1.,  0., 0.)),

             'blue':  ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0))}
    cmap = colors.LinearSegmentedColormap('custom', cdict, N=n)
    return cmap


def get_blue_map(n):
    cdict = {'red':   ((0.0,  0.1, 0.1),
                       (1.,  0., 0.)),

             'green': ((0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)),

             'blue':  ((0.0,  0.3, 0.3),
                       (1.0,  1.0, 1.0))}
    cmap = colors.LinearSegmentedColormap('custom', cdict, N=n)
    return cmap

def colorize(z, theme = 'dark', saturation = 1., beta = 1.4, transparent = False, alpha = 1., max_threshold = 1):
    r = np.abs(z)
    r /= max_threshold*np.max(np.abs(r))
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'white' else 1.- 1./(1. + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = np.transpose(c, (1,2,0))  
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def show_colormap_image(to_img, save_name = None):

    n_phi = 30
    n_amp = 100
    X,Y = np.meshgrid(np.linspace(-np.pi,np.pi,n_phi),np.linspace(1.,0.,n_amp))

    cm = Y*np.exp(1j*X)

    fig, ax = plt.subplots(1,1)
    img = ax.imshow(to_img(cm.transpose()), extent = [-np.pi, np.pi, 0., 1.], aspect = 30)

    ax.set_xticks([-np.pi,0,np.pi])
    ax.set_xticklabels([r'$\pi$', '0', r'$\pi$'])
    ax.set_yticks([0,0.5,1])
    if save_name:
        plt.savefig(save_name, dpi = 200)