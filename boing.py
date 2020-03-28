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

def Cross(ax, ay, az, bx, by, bz):
    return ay*bz-by*az, az*bx-bz*ax, ax*by-bx*ay

def PointOnLine(lmbda, A, B):
    return A[0]+lmbda*B[0], A[1]+lmbda*B[1], A[2]+lmbda*B[2]

class Light:
    def __init__(self, location):
        self.location=location

    def illuminate(self, x, y, z, nx, ny, nz):
        # d
        dx=self.location[0]-x
        dy=self.location[1]-y
        dz=self.location[2]-z
        magd2=dx**2+dy**2+dz**2
        magd=np.sqrt(magd2)
        magn=np.sqrt(nx**2+ny**2+nz**2)

        # alpha = angle between normal to surface (outwards) and line from surface to L
        cosalpha=(nx*dx+ny*dy+nz*dz)/(magd*magn)

        # Illumination of surface by light 1/R2 law and projection effect of angle
        lfac=cosalpha/magd2
        lfac/=np.nanmax(lfac) # Normalise so max is 1

        # If normal not pointing towards light then no illumination
        lfac[lfac<0]=0

        return lfac

class Ground:
    def __init__(self, point, normal, dx=3, dy=10):
        self.o=point # Ground passes through this point
        self.n=normal/np.sqrt(np.sum(normal**2)) # Unit normal to ground
        self.dx=dx
        self.dy=dy

    def check(self, camera_location, Rx, Ry, Rz):
        # Lambda for where ray intersects ground
        lambda_ground=np.sum((self.o-camera_location)*self.n)/(Rx*self.n[0]+Ry*self.n[1]+Rz*self.n[2])

        # If ray intersects ground
        intersects_ground=(lambda_ground > 0) & np.isfinite(lambda_ground)

        return intersects_ground, lambda_ground

    def image(self, ground_points, light, intersects_ground_first):
        # ground_points = points of intersection of rays with ground
        Gx, Gy, Gz=ground_points

        lfac=light.illuminate(Gx, Gy, Gz, self.n[0], self.n[1], self.n[2]) 

        ix=np.floor(Gx/self.dx).astype(int)
        iy=np.floor(Gy/self.dy).astype(int)
        ground=intersects_ground_first*np.mod(ix+iy, 2)*lfac

        ground_image=ImageToRGB(ground, rgb=[0, 0.3, 0], rgb0=[0, 0, 0])
        return ground_image

    def reflect(self, R1x, R1y, R1z):
        R2_perp_mag=-(R1x*self.n[0]+R1y*self.n[1]+R1z*self.n[2])
        R2_perp_x=R2_perp_mag*self.n[0]
        R2_perp_y=R2_perp_mag*self.n[1]
        R2_perp_z=R2_perp_mag*self.n[2]
        R1_mag=R1x**2+R1y**2+R1z**2
        R1_cross_n=Cross(R1x/R1_mag, R1y/R1_mag, R1z/R1_mag, self.n[0], self.n[1], self.n[2])
        R1_cross_n_cross_n=Cross(R1_cross_n[0], R1_cross_n[1], R1_cross_n[2], self.n[0], self.n[1], self.n[2])
        R2_parallel_mag=(R1x*R1_cross_n_cross_n[0]+R1y*R1_cross_n_cross_n[1]+R1z*R1_cross_n_cross_n[2])
        R2_parallel_x=R2_parallel_mag*R1_cross_n_cross_n[0]
        R2_parallel_y=R2_parallel_mag*R1_cross_n_cross_n[1]
        R2_parallel_z=R2_parallel_mag*R1_cross_n_cross_n[2]
        R2=R2_perp_x+R2_parallel_x, R2_perp_y+R2_parallel_y, R2_perp_z+R2_parallel_z
        return R2

class Ball:
    def __init__(self, centre, radius, dphi=np.pi/10, dtheta=np.pi/10):
        self.centre=centre
        self.radius=radius
        self.dphi=dphi
        self.dtheta=dtheta

    def check(self, camera_location, Rx, Ry, Rz):
        BC=camera_location[0]-self.centre[0], camera_location[1]-self.centre[1], camera_location[2]-self.centre[2]

        wok=np.where(np.isfinite(Rx+BC[0]+Ry+BC[1]+Rz+BC[2]))
        RdotBC=Rx*BC[0]+Ry*BC[1]+Rz*BC[2]

        # P = point of closest approach of ray to B
        BPx=BC[0]-RdotBC*Rx
        BPy=BC[1]-RdotBC*Ry
        BPz=BC[2]-RdotBC*Rz
        magBP=BPx**2+BPy**2+BPz**2

        # If ray intersects ball
        #intersects_ball=magBP <= self.radius**2
        #print(intersects_ball.shape)

        # Lowest of two lambda values for location where rays intersect with surface of ball
        #beta=np.sqrt(RdotBC**2-np.linalg.norm(BC)**2+self.radius**2) # Second term of quadratic eqn soln
        beta=np.sqrt(RdotBC**2-(BC[0]**2+BC[1]**2+BC[2]**2)+self.radius**2) # Second term of quadratic eqn soln
        #print(beta.shape)
        lambda_ball=-RdotBC-beta
        #if not np.array_equal(intersects_ball[wok], np.isfinite(beta[wok])):
        #    print(np.isfinite(beta).astype(int)-intersects_ball.astype(int))
        #    print(lambda_ball)
        #    raise RuntimeError("Expected intersections tests to be identical") # Sanity check
        intersects_ball=np.isfinite(lambda_ball) & (lambda_ball>0)
        #print(intersects_ball)

        return intersects_ball, lambda_ball

    def image(self, S_points, light, intersects_ball_first):
        # S_points = first (closest to camera) points where ray intersects surface of ball
        Sx, Sy, Sz=S_points

        # BS
        SBx=Sx-self.centre[0]
        SBy=Sy-self.centre[1]
        SBz=Sz-self.centre[2]
        phi=np.arctan2(SBy, SBx)
        theta=np.arctan2(SBz, np.sqrt(SBx**2+SBy**2))

        lfac=light.illuminate(Sx, Sy, Sz,
                              SBx,
                              SBy,
                              SBz)

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

    light=Light(np.array([10, 130, 5]))
    #light=Light(np.array([10, 130, 100]))
    #light=Light(np.array([100, 130, 100]))
    #ground=Ground(point=np.array([0, 0, -5]), normal=np.array([0, 0, 1]), dx=3, dy=10)
    ground=Ground(point=np.array([0, 0, -5]), normal=np.array([0, 0, 1]), dx=1, dy=2)
    ball=Ball(centre=np.array([0, 150, -1.5]), radius=5)
    ground_reflectivity=0.2
    camera_location=np.array([0, 0, 0]) # Camera location

    w=500 # Image pixel grid width
    h=500 # Image pixel grid height
    #w=16
    #h=16
    thetaHr=np.array([-1, 1])*7/100 # Horizontal image physical range (radians) 
    thetaVr=np.array([-1, 1])*7/100 # Vertical image physical range (radians) 
    thetaH, thetaV=ImageGrid(w, h, xr=thetaHr, yr=thetaVr) # Angle at each image location (radians)

    tan_thetaH=np.tan(thetaH)
    tan_thetaV=np.tan(thetaV)

    # Ray for each image location has eqn r = camera_location + lambda * [Rx, Ry, Rz]
    Ry=np.sqrt(1/(1+tan_thetaH**2+tan_thetaV**2))
    Rx=Ry*tan_thetaH
    Rz=Ry*tan_thetaV

    ground_intersects, ground_lambda=ground.check(camera_location, Rx, Ry, Rz)
    ball_intersects, ball_lambda=ball.check(camera_location, Rx, Ry, Rz)

    intersects_ground_only=ground_intersects & np.logical_not(ball_intersects)
    intersects_ball_only=ball_intersects & np.logical_not(ground_intersects)
    intersects_ball_and_ground=ball_intersects & ground_intersects
    intersects_ball_first=intersects_ball_only | (intersects_ball_and_ground & (ball_lambda <= ground_lambda))
    intersects_ground_first=intersects_ground_only | (intersects_ball_and_ground & (ball_lambda > ground_lambda))

    Rref=ground.reflect(Rx, Ry, Rz) # Camera ray directions reflected by the ground

    ground_points=PointOnLine(ground_lambda, camera_location, [Rx, Ry, Rz])
    ground_reflects_ball, ground_ball_reflection_lambda=ball.check(ground_points, Rref[0], Rref[1], Rref[2])
    ground_ball_reflection_points=PointOnLine(ground_ball_reflection_lambda, ground_points, Rref)

    ground_image=ground.image(ground_points,
                              light, intersects_ground_first)
    ball_image=ball.image(PointOnLine(ball_lambda, camera_location, [Rx, Ry, Rz]),
                          light, intersects_ball_first)
    ground_image[ground_reflects_ball]*=(1.-ground_reflectivity)
    ground_ball_reflection_image=ball.image(ground_ball_reflection_points, light, ground_reflects_ball)

    # Plot it
    size_cm=(29.7, 21.0)
    fig=plt.figure(figsize=(size_cm[0]/2.54, size_cm[1]/2.54))
    ax=fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("BoingTrace - Python/NumPy toy ray tracer")
    ax.imshow(ball_image+ground_image+ground_ball_reflection_image*ground_reflectivity,
              interpolation='nearest',
              extent=[thetaHr[0], thetaHr[1], thetaVr[0], thetaVr[1]],
              origin='lower')


    plt.show()

if __name__ == "__main__":

    #Example()
    Boing()
