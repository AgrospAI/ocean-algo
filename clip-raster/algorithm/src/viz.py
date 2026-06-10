import os
import matplotlib.pyplot as plt

def draw_raster(img, name: str, cmap_: str):
    os.makedirs('figures/rasters', exist_ok=True)
    save_path = f'figures/rasters/{name}.png'

    plt.imshow(img, cmap = cmap_) # vmin - 1, vmax 2
    plt.colorbar()
    plt.title(f'{name}')
    
    plt.savefig(save_path)
    plt.close()