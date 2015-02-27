import argparse
import os
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import numpy as np
from math import pow
import pylab as plt
from astroML.plotting import scatter_contour
from astroML.plotting.tools import draw_ellipse
from astroML.density_estimation import EmpiricalDistribution
from sklearn.mixture import GMM
import galsim

def nz(z0, z):
    '''return the N(z)'''
    nz = z*z*np.exp(-z/z0)
  
    return nz

def selectRedshift (mag):
    '''Given a magnitude and an N(z) draw a random redshift
    
    Uses Wilson\'s approximation for n(z)=0.5z^2exp(-z/zo)/zo^3

    assumes I band we will use i band selection
    '''
    
    # calc zo from linear fit to wilsons data (subtract 0.5 to get in r band)
    mag = mag - 0.5
    if (mag > 19.):
        zo = 0.1000000015*(mag) -1.885
    else:
        zo = 0.005

    maxnz = nz(zo,2.*zo)

    # randomly select in z and nz
    _nz = 1000.
    nzfit = 0.
    while (_nz >= nzfit):
        _z = 5. * np.random.rand()
        nzfit = nz(zo,_z)
        _nz = maxnz*np.random.rand()

    return _z

def nm(m, slope=0.4):
    '''Number magnitude relation for galaxies assuming a fixed number count slope'''

    nm = pow(10., slope*m)
    return nm


def selectMagnitude(magLim):
    '''Randomly select galaxy magnitudes'''
    maxnm = nm(magLim)
  
    _nm = 10000000000.
    nmfit = 0.;
    while (_nm > nmfit):
        _m = 17.+ (magLim-17.)*np.random.rand()
        _nm = maxnm*np.random.rand()
        nmfit = nm(_m)
 
    return _m;



class Catalog():
    '''Catalog class to hold input parameters for galaxies

    columnNames = ['objID', 'z', 'Sp', 'Scale', 'Vmax', 'gg2d', 'e_gg2d', 'rg2d', 'e_rg2d',
    '__B_T_g', 'e__B_T_g', '__B_T_r', 'e__B_T_r', 'Rhlg', 'Rhlr', 'Re', 'e', 'e_e', 'Rd',
    'e_Rd', 'i', 'e_i', 'phid', 'e_phid', 'S2g', 'S2r', 'ggMag', 'gbMag', 'gdMag',
    'rgMag', 'rbMag', 'rdMag', 'nb', 'All', 'Sloan', 'DR7']
    '''
    
    def __init__(self, filename):
        self.readFitsFile(filename)
        
    def readFitsFile (self, filename, dataHDU=1):
        '''Read in catalog from fits file'''
        hdulist = fits.open(filename)
        self.table = hdulist[dataHDU].data
        self.columnNames = hdulist[dataHDU].columns.names
        hdulist.close()

    def selectColumns (self, selectColumns=['objID', 'z', 'rg2d', '__B_T_r', 'nb', 'Re',
                                            'e', 'Rd', 'i', 'phid', 'Scale'], maxPts=None):
        '''Given a set of columns return a recarray with data

        read only the first maxPts
        '''
        if (maxPts == None):
            data = np.zeros((len(self.table),len(selectColumns)))
            for i,name in enumerate(selectColumns):
                data[:,i] = self.table[name]
        else:
            data = np.zeros((maxPts, len(selectColumns)))
            for i,name in enumerate(selectColumns):
                data[:maxPts,i] = self.table[name][:maxPts]
            
        return data


class GalaxyProperties():
    '''Class that stores the probabilities of galaxy parameters'''
    def mixtureModel(self, nGaussians,n_iter=1000, min_covar=3, covariance_type='full'):
        '''Define the mixture model'''
        self.nGaussians = nGaussians
        self.clf = GMM(nGaussians, covariance_type=covariance_type,
                       n_iter=n_iter, min_covar=min_covar, random_state=0)

    def learnModel(self, data):
        '''Fit the mixture model given a set of data'''
        self.clf.fit(data)
        print("converged:", self.clf.converged_)

    def selectRandomND(self, nGals):
        '''Select nGals at random from the density plot and return parameters'''
        return self.clf.sample(5000)

    def selectRandom1D(self, data, nrandom):
        '''Select select a random sample based on a 1D distribution'''
        return EmpiricalDistribution(data).rvs(nrandom)

    def writeModel(self, filename):
        '''Write model density to file'''

    def readModel(self, filename):
        '''read model density from file'''


def plotPairwise(data, fig, labels=None, mixtures=None, limits=None, **kwargs):
    '''Plot a set of pairwise correlations'''
    
    nrow, ncol = data.shape
    if labels is None:
        labels = ['var%d'%i for i in range(ncol)]

    for i in range(ncol):
        for j in range(ncol):
            nSub = i * ncol + j + 1
            ax = fig.add_subplot(ncol, ncol, nSub)
            if i == j:
                ax.hist(data[:,i], bins=100)
                if (limits != None):
                    ax.set_xlim(limits[i])
            else:
                scatter_contour(data[:,i], data[:,j], threshold=200, log_counts=True, ax=ax,
                                histogram2d_args=dict(bins=20),
                                plot_args=dict(marker=',', linestyle='none', color='black'),
                                contour_args=dict(cmap=plt.cm.bone))

                # plt fit ellipses
                for k in range(mixtures.n_components):
                    mean = mixtures.means_[k][[i,j]]
                    cov = mixtures.covars_[k][[i,j]][:,[i,j]]
                    if cov.ndim == 1:
                        cov = np.diag(cov)
                    draw_ellipse(mean, cov, ax=ax, fc='none', ec='k', zorder=2, scales=[1])
                if (limits != None):
                    ax.set_xlim(limits[j])
                    ax.set_ylim(limits[i])
            if (i==0):
                ax.set_title(labels[j])
            if (j==0):
                ax.set_ylabel(labels[i])

def plotHistogram(data, fig, index=None, labels=None, **kwargs):
    '''Plot a histogram of a data set'''

    if labels is None:
        labels = ['var1']

    ax = fig.add_subplot(111,**kwargs)
    if (index != None):
        ax.hist(data[:,index], bins=50)
    else:
        ax.hist(data, bins=100)
    ax.set_title(labels)

def setBounds(value, minVal, maxVal):
    '''Set upper and lower bounds for a variable'''
    if (value < minVal):
        value = minVal
    if (value > maxVal):
        value = maxVal
    return value
    
def main():
    '''Driver for generating galaxy images for the SDSS

    Input parameters are set by sampling from gim2D fits to SDSS images

    Input images are generated using galsim
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFile", type=str, default="asu.fit.gz",
                        help="Galaxy properties file")
    parser.add_argument("--outputDirectory", type=str, default='output',
                        help="Output directory of images")
    parser.add_argument("--outputFile", type=str, default='galaxies.dat',
                        help="Output file with properties of images")
    parser.add_argument("--maxPts", type=int, default=100000,
                        help="Max number of points to read from input file")
    parser.add_argument("--nRandom", type=int, default=100,
                        help="Number of random galaxies to generate")
    parser.add_argument("--nGaussians", type=int, default=5,
                        help="Number of gaussians to fit density of galaxy properties")
    parser.add_argument("--seed", type=int, default=1827493,
                        help="Random seed")
    parser.add_argument("--csv", type=bool, default=False,
                        help="Save images as csv files")
    args = parser.parse_args()

    
    columns = ['Re', 'Rd', '__B_T_r', 'e', 'i', 'phid',]

    #read and select data columns
    maxPts = args.maxPts
    catalog = Catalog(args.inputFile)
    data = np.nan_to_num(catalog.selectColumns(columns, maxPts=maxPts))

    # create density estimation of galaxy properties that are correlated
    properties = GalaxyProperties()
    properties.mixtureModel(args.nGaussians)
    print properties.clf
    #TODO - select columts for 1D and 2D fits from the command line
    properties.learnModel(data[:,[0,1]])

    # plot data and model as an NxN pairwise plot and model
    fig = plt.figure(figsize=(10, 10))
    plotPairwise(data[:,[0,1]], fig, labels=columns, mixtures=properties.clf,
                 limits=[[-1,20],[-1,20]])

    #draw a random data set with these properties
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    nsample = args.nRandom
    m = np.zeros(nsample)
    z = np.zeros(nsample)

    # select magnitude, redshift, and kpc/arcsec TODO - replace this with a sampling method
    for i in xrange(nsample):
        m[i] = selectMagnitude(22.5)
        z[i] = selectRedshift(m[i])
    sizeScale = cosmo.kpc_proper_per_arcmin(z)/60. 

    # sample uncorrelated data
    BTT = properties.selectRandom1D(data[:,2], nsample)
    e = properties.selectRandom1D(data[:,3], nsample)
    i = properties.selectRandom1D(data[:,4], nsample)
    phi = properties.selectRandom1D(data[:,5], nsample)


    # sample correlated data
    sampled_data = properties.selectRandomND(nsample)

    # create galsim images
    pixel = 0.45
    random_seed = args.seed  
    rng = galsim.BaseDeviate(random_seed)
    of = open(os.path.join(args.outputDirectory,args.outputFile),'w')
#    of.write("#Index, m, z, counts, Re(arcsec), Rd(arsec), BTT, ellipticity_bulge, inclination_disk, PA_bulge, PA_disk\n") 
    of.write("#Index, m, z, mu_x, mu_y, counts_bulge, counts_disk, Re(pixels), Rd(pixels), ellipticity_bulge, inclination_disk, PA\n") 
    for i,(mGal,zGal,scaleGal,eGal,iGal,phiGal,bttGal,(Re,Rd)) \
            in enumerate(zip(m,z,sizeScale,e, i, phi, BTT, sampled_data)):
        #set bounds for values
        print mGal,zGal,scaleGal,eGal,iGal,phiGal,bttGal,Re,Rd
        ReScale = setBounds(Re/scaleGal.value, 0., 10.)
        RdScale = setBounds(Rd/scaleGal.value, 0., 10.)
        bttGal = setBounds(bttGal, 0., 1.)
        eGal = setBounds(eGal, 0., 1.)


#        print i,mGal,zGal,ReScale,RdScale
        bulge = galsim.Sersic(4, half_light_radius=ReScale)
        shear = galsim.Shear(q=1.-eGal,beta=phiGal*galsim.radians)
        bulge = bulge.shear(shear)
        disk = galsim.Sersic(1., scale_radius=RdScale)
        q0 = 0.2
        q = np.sqrt(((1-q0**2) * np.cos(np.radians(iGal))**2)  + q0**2)
        shear = galsim.Shear(q=q, beta=phiGal*galsim.radians)
        disk = disk.shear(shear)
        gal = bttGal * bulge + (1. - bttGal) * disk
        # TODO - fix to derive from SDSS images
        counts =  10**((mGal - 20.)/-2.5)*1.91966000000000E+03
        gal = gal.withFlux(counts)
        # TODO - fix to derive from SDSS PSF
        psf = galsim.Gaussian(flux=1., sigma=0.5) # PSF flux should always = 1
        final = galsim.Convolve([psf, gal])
        img = galsim.ImageF(64, 64, scale=pixel)
        image = final.drawImage(image=img)
        image.write('%s/testImage_%d.fits'%(args.outputDirectory,i))
        if (args.csv == True):
            np.savetxt('%s/testImage_%d.csv.gz'%(args.outputDirectory,i), img.array, delimiter=",")
        of.write("%d, %g, %g, 32., 32., %g, %g, %g, %g, %g, %g, %g %g\n"
                 %(i,mGal,zGal,counts*bttGal,counts*(1.-bttGal),ReScale/pixel,RdScale/pixel,eGal,iGal,phiGal, phiGal))
        
    plt.show()
    

if __name__ == '__main__':
    main()
