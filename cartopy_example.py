import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()


# Downloaded from http://biogeo.ucdavis.edu/data/gadm2/shp/DEU_adm.zip
fname = 'shape_file/united_kingdom/gadm36_GBR_0.shp'

adm1_shapes = list(shpreader.Reader(fname).geometries())

ax = plt.axes(projection=ccrs.PlateCarree())

# plt.title('United Kingdom')

ax.coastlines(resolution='10m')

ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
                  edgecolor='black', facecolor='gray', alpha=0.5)

ax.set_extent([-8.1, 2, 48, 62], ccrs.PlateCarree())

X, Y = np.meshgrid(np.arange(-8.1, 2, 0.2), np.arange(48, 62 * np.pi, 0.2))
U = np.log(X + 8.2)
V = np.sin(Y)

Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               pivot='mid', scale=10, units='inches', width=0.01, color='red')
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')

plt.axis('off')
plt.savefig('map_plot.png', bbox_inches='tight')
