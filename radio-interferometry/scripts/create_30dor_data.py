import numpy as np
import os
import astropy
import simms
import glob
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from casatasks import simobserve, importfits
from casatools import componentlist, image, measures, simulator
from multiprocessing import Pool

import shutil
import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['LD_LIBRARY_PATH']}:/home/mars/miniforge3/lib/"

def generate_random_empty_ms(ms_name = 'empty.ms'):
    """creates empty meerkat measurement set"""
    ra = np.random.rand()*360
    dec = -30 + np.random.rand()*80 - 40
    coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
    direction = coord.to_string('hmsdms',precision=2).replace(' ',',')
    f0 = int(800 + 700*np.random.rand())
    simms_call = f"""simms -dir J2000,{direction} -os -T meerkat -dt 240 -st 8 -nc 10 -f0 {f0}MHz -df 10MHz -pl XX YY -n {ms_name}"""
    os.system(simms_call)

def create_mock_image(ms_name, image_name, im_size):
    """creates psf and empty reconstructed image"""
    wsclean_call = f''' wsclean -name {image_name} -size {im_size} {im_size} -scale 2asec -weight briggs 0.5 -make-psf {ms_name} '''
    os.system(wsclean_call)

def create_model_image(image_name, image):
    """Fill empty image with target image"""
    f = fits.open(f"{image_name}-image.fits", mode='readonly')
    f[0].data = image[None, None]
    f.writeto(f"{image_name}-model.fits", overwrite=True)
    f.close()

def simulate_observation(ms_name, image_name, sigma='1.4mJy'):
    importfits( fitsimage = f'{image_name}-model.fits', imagename =  f'{image_name}-model.image', overwrite=True)    
    sm.openfromms(ms_name)
    sm.predict( imagename = f'{image_name}-model.image')
    sm.setnoise(mode = 'tsys-atm')
    sm.corrupt()
    sm.done()

def create_dirty_image(ms_name, image_name, im_size):
    wsclean_call = f''' wsclean -name {image_name} -size {im_size} {im_size} -scale 2asec -weight briggs 0.5 {ms_name} '''
    os.system(wsclean_call)

def check_im(image_name):
    plt.imshow(np.squeeze(fits.getdata(f"{image_name}-model.fits")))
    plt.colorbar()
    plt.show()
    plt.imshow(np.squeeze(fits.getdata(f"{image_name}-dirty.fits")))
    plt.colorbar()
    plt.show()


data_folder = "/share/gpu0/mars/TNG_data/processed_360"
# im_paths = glob.glob(data_folder + "/*.npy")

im_paths = [
    "/home/mars/src_aiai/notebooks/30Dor_360.npy"
]

sm = simulator()

def init():
    global rng
    rng = np.random.default_rng()

def task(j):
    print("="*10, j)
    i=0
    directory = f"/share/gpu0/mars/im_{j%100}"
    os.makedirs(directory, exist_ok=True)

    ms_name = f"{directory}/im_{i}.ms"
    image_name=f"{directory}/im_{i}"
    im_size = 360

    image = np.load(im_paths[j])
    print(image.max())
    image = image/image.max()
    

    done = False
    tries = 0

    generate_random_empty_ms(ms_name)
    create_mock_image(ms_name, image_name, im_size)

    points = np.zeros_like(image)
    create_model_image(image_name, image)
    simulate_observation(ms_name, image_name)
    create_dirty_image(ms_name, image_name, im_size)

    dirty = np.squeeze(fits.getdata(f"{image_name}-dirty.fits"))
    psf = np.squeeze(fits.getdata(f"{image_name}-psf.fits"))

    return image, dirty, psf

res = task(0)
np.save('/home/mars/src_aiai/notebooks/GAN_30Dor.npy', res)