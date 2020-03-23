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

class Ground:
    def __init__(self, point, normal, dx=3, dy=10):
        self.o=point # Ground passes through this point
        self.n=normal # Unit normal to ground
        self.dx=dx
        self.dy=dy

    def check(self, C, Rx, Ry, Rz):
        # Lambda for where ray intersects ground
        lambda_ground=np.sum((self.o-C)*self.n)/(Rx*self.n[0]+Ry*self.n[1]+Rz*self.n[2])

        # If ray intersects ground
        intersects_ground=(lambda_ground > 0) & np.isfinite(lambda_ground)

        self.intersects=intersects_ground
        self.lmbda=lambda_ground

    def image(self, C, Rx, Ry, Rz, L, intersects_ground_first):
        # Point of intersection with ground
        Gx=C[0]+self.lmbda*Rx
        Gy=C[1]+self.lmbda*Ry
        Gz=C[2]+self.lmbda*Rz

        # GL
        GLx=L[0]-Gx
        GLy=L[1]-Gy
        GLz=L[2]-Gz
        magGL2=GLx**2+GLy**2+GLz**2
        magGL=np.sqrt(magGL2)

        # alpha = angle between normal to surface (outwards) and line from surface to L
        cosalpha=-(self.n[0]*GLx+self.n[1]*GLy+self.n[2]*GLz)/magGL

        # Illumination of surface by light 1/R2 law and projection effect of angle
        lfac=cosalpha/magGL2
        lfac/=np.nanmax(lfac) # Normalise so max is 1

        # If normal not pointing towards light then no illumination
        lfac[lfac<0]=0

        ix=np.floor(Gx/self.dx).astype(int)
        iy=np.floor(Gy/self.dy).astype(int)
        ground=intersects_ground_first*np.mod(ix+iy, 2)*lfac

        ground_image=ImageToRGB(ground, rgb=[0, 0.3, 0], rgb0=[0, 0, 0])
        return ground_image

class Ball:
    def __init__(self, centre, radius, dphi=np.pi/10, dtheta=np.pi/10):
        self.centre=centre
        self.radius=radius
        self.dphi=dphi
        self.dtheta=dtheta

    def check(self, C, Rx, Ry, Rz):
        BC=C-self.centre

        RdotBC=Rx*BC[0]+Ry*BC[1]+Rz*BC[2]

        # P = point of closest approach of ray to B
        BPx=BC[0]-RdotBC*Rx
        BPy=BC[1]-RdotBC*Ry
        BPz=BC[2]-RdotBC*Rz
        magBP=BPx**2+BPy**2+BPz**2

        # If ray intersects ball
        intersects_ball=magBP <= self.radius**2

        # Lowest of two lambda values for location where rays intersect with surface of ball
        beta=np.sqrt(RdotBC**2-np.linalg.norm(BC)**2+self.radius**2) # Second term of quadratic eqn soln
        if not np.array_equal(intersects_ball, np.isfinite(beta)):
            raise RuntimeError("Expected intersections tests to be identical") # Sanity check
        lambda_ball=-RdotBC-beta

        self.intersects=intersects_ball
        self.lmbda=lambda_ball

    def image(self, C, Rx, Ry, Rz, L, intersects_ball_first):
        # S = first (closest to camera) location where ray intersects surface of ball
        Sx=C[0]+self.lmbda*Rx
        Sy=C[1]+self.lmbda*Ry
        Sz=C[2]+self.lmbda*Rz

        # SL
        SLx=Sx-L[0]
        SLy=Sy-L[1]
        SLz=Sz-L[2]

        # SB
        SBx=Sx-self.centre[0]
        SBy=Sy-self.centre[1]
        SBz=Sz-self.centre[2]
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

        # Pattern on ball
        iphi=np.floor(phi/self.dphi).astype(int)
        itheta=np.floor(theta/self.dtheta).astype(int)
        w_noball=np.where(np.logical_not(intersects_ball_first))
        red_ball=intersects_ball_first*np.mod(iphi+itheta, 2)*lfac
        red_ball[w_noball]=0
        white_ball=intersects_ball_first*np.mod(iphi+itheta+1, 2)*lfac
        white_ball[w_noball]=0

        red_ballimage=ImageToRGB(red_ball, rgb=[1, 0, 0], rgb0=[0, 0, 0])
        white_ballimage=ImageToRGB(white_ball, rgb=[1, 1, 1], rgb0=[0, 0, 0])

        return red_ballimage+white_ballimage

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
    L=np.array([5, 130, 5]) # Point source light

    ground=Ground(point=np.array([0, 0, -5]), normal=np.array([0, 0, 1]))
    ball=Ball(centre=np.array([0, 150, -1.5]), radius=5)

    w=2000
    h=2000

    #w=160
    #h=160
    thetaHr=np.array([-1, 1])*5/100
    thetaVr=np.array([-1, 1])*5/100
    thetaH, thetaV=ImageGrid(w, h, xr=thetaHr, yr=thetaVr) # Angle at each image location (radians)

    #zoom_sigma=3
    #zoom_amp=3
    #thetaH, thetaV+=zoom_amp*np.exp(-(thetaH**2+thetaV**2)/2*zoom_sigma^2)

    tan_thetaH=np.tan(thetaH)
    tan_thetaV=np.tan(thetaV)

    # Ray for each image location has eqn r = C + lambda * [Rx, Ry, Rz]
    Ry=np.sqrt(1/(1+tan_thetaH**2+tan_thetaV**2))
    Rx=Ry*tan_thetaH
    Rz=Ry*tan_thetaV

    ground.check(C, Rx, Ry, Rz)
    ball.check(C, Rx, Ry, Rz)

    intersects_ground_only=ground.intersects & np.logical_not(ball.intersects)
    intersects_ball_only=ball.intersects & np.logical_not(ground.intersects)
    intersects_ball_and_ground=ball.intersects & ground.intersects
    intersects_ball_first=intersects_ball_only | (intersects_ball_and_ground & (ball.lmbda <= ground.lmbda))
    intersects_ground_first=intersects_ground_only | (intersects_ball_and_ground & (ball.lmbda > ground.lmbda))

    ground_image=ground.image(C, Rx, Ry, Rz, L, intersects_ground_first)
    ball_image=ball.image(C, Rx, Ry, Rz, L, intersects_ball_first)


    ax.imshow(ball_image+ground_image,
              interpolation='nearest',
              extent=[thetaHr[0], thetaHr[1], thetaVr[0], thetaVr[1]],
              origin='lower')


    plt.show()

if __name__ == "__main__":

    #Example()
    Boing()
