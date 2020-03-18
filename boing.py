#! /usr/bin/env python3

"""
Tools for common plotting actions
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

"""Return x and y images corresponding to mid-pixel positions of image of width w, height h covering ranges xr and yr. Default xr=[0, w], yr=[0, h]"""
def ImageGrid(w, h, xr=None, yr=None):
    if xr is None: xr=[0, w]
    if yr is None: yr=[0, h]
    y, x=np.mgrid[0:h, 0:w]
    x=(x+0.5)/w*(xr[1]-xr[0])+xr[0]
    y=(y+0.5)/h*(yr[1]-yr[0])+yr[0]
    return x, y

"""Create RGB image with colour set to rgb0+im*(rgb-rgb0)"""
def ImageToRGB(im, rgb0=[0, 0, 0], rgb=[0, 1, 0]):
    maxfac=im.max() if im.max()>1 else 1 # Limit maximum to <=1
    h, w=im.shape
    imrgb=np.zeros((h, w, 3))
    for icomp in range(3):
        imrgb[:, :, icomp]+=rgb0[icomp]+im*(rgb[icomp]-rgb0[icomp])
    return imrgb/maxfac

def Boing():

    size_cm=(29.7, 21.0)
    fig=plt.figure(figsize=(size_cm[0]/2.54, size_cm[1]/2.54))
    ax=fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)

    #
    #  Z
    #  |
    #  |      Y
    #  |     /
    #  |    /
    #  |   /
    #  |  /
    #  | /
    #  |/_______________ X
    #

    C=np.array([0, 0, 0]) # Camera location
    B=np.array([0, 150, 0]) # Centre of ball
    #L=np.array([0, 150, 50]) # Point source light
    #L=np.array([50, 50, 50]) # Point source light
    L=np.array([5, 145, 5]) # Point source light
    #B=np.array([3, 10, 4]) # Centre of ball
    ball_radius=5

    w=500
    h=500
    #w=10
    #h=10
    thetaHr=np.array([-1, 1])*5/100
    thetaVr=np.array([-1, 1])*5/100
    thetaH, thetaV=ImageGrid(w, h, xr=thetaHr, yr=thetaVr) # Angle at each image location (radians)

    tan_thetaH=np.tan(thetaH)
    tan_thetaV=np.tan(thetaV)

    # Ray for each image location has eqn r = C + lambda * [Rx, Ry, Rz]
    Ry=np.sqrt(1/(1+tan_thetaH**2+tan_thetaV**2))
    Rx=Ry*tan_thetaH
    Rz=Ry*tan_thetaV

    BC=C-B

    RdotBC=Rx*BC[0]+Ry*BC[1]+Rz*BC[2]

    # P = point of closest approach of ray to B
    BPx=BC[0]-RdotBC*Rx
    BPy=BC[1]-RdotBC*Ry
    BPz=BC[2]-RdotBC*Rz
    magBP=BPx**2+BPy**2+BPz**2
    intersects=magBP <= ball_radius**2

    # Lowest of two lambda values for location where rays intersects with surface of ball
    beta=np.sqrt(RdotBC**2-np.linalg.norm(BC)**2+ball_radius**2) # Second term of quadratic eqn soln
    if not np.array_equal(intersects, np.isfinite(beta)):
        raise RuntimeError("Expected intersections tests to be identical") # Sanity check
    lambda_intersect=-RdotBC-beta

    # S = first (closest to camera) location where ray intersects surface of ball
    Sx=C[0]+lambda_intersect*Rx
    Sy=C[1]+lambda_intersect*Ry
    Sz=C[2]+lambda_intersect*Rz

    # SL
    SLx=Sx-L[0]
    SLy=Sy-L[1]
    SLz=Sz-L[2]

    # SB
    SBx=Sx-B[0]
    SBy=Sy-B[1]
    SBz=Sz-B[2]
    phi=np.arctan2(SBy, SBx)
    theta=np.arctan2(SBz, np.sqrt(SBx**2+SBy**2))

    # Normal to surface at S
    magSB=np.sqrt(SBx**2+SBy**2+SBz**2)
    magSL2=SLx**2+SLy**2+SLz**2
    magSL=np.sqrt(magSL2)

    # alpha = angle between normal to surface (outwards) and line from surface to L
    cosalpha=-(SBx*SLx+SBy*SLy+SBz*SLz)/(magSB*magSL)

    # Illumination of surface by light 1/R2 law and projection effect of angle
    lfac=cosalpha/magSL2
    lfac/=np.nanmax(lfac) # Normalise so max is 1

    # If normal not pointing towards light then no illumination
    lfac[lfac<0]=0

    dphi=np.pi/10
    dtheta=np.pi/10
    iphi=np.floor(phi/dphi).astype(int)
    itheta=np.floor(theta/dtheta).astype(int)
    red=intersects*np.mod(iphi+itheta, 2)*1.*lfac
    redimage=ImageToRGB(red, rgb=[1, 0, 0], rgb0=[0, 0, 0])
    white=intersects*np.mod(iphi+itheta+1, 2)*1.*lfac
    whiteimage=ImageToRGB(white, rgb=[1, 1, 1], rgb0=[0, 0, 0])
    bg=np.zeros_like(intersects)
    bg[np.where(np.logical_not(intersects))]=0.5
    bgimage=ImageToRGB(bg, rgb=[0, 1, 0], rgb0=[0, 0, 0])
    ax.imshow(redimage+whiteimage+bgimage,
              interpolation='nearest',
              extent=[thetaHr[0], thetaHr[1], thetaVr[0], thetaVr[1]],
              origin='lower')


    plt.show()

if __name__ == "__main__":

    #Example()
    Boing()
