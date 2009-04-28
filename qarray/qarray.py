#!/usr/bin/env python
"""
-------------------------------------------------------------------------------
qarray
Copyright (C) William Dickson, 2008.
  
wbd@caltech.edu
www.willdickson.com

Released under the LGPL Licence, Version 3

This file is part of qarray.

qarray is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
      
qarray is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with qarray.  If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------   
qarray.py 

Purpose: This module provides a simple set of functions for dealing with
arrays of quaternions. It includes functions for multipying, finding the
conjugate, magnitudes, and inverses of arrays of quaternions. In addition
utility routines are provided for converitng vector and euler angle arrays
to arrays of quaternions. 

Functions:

    qarray_multiply       - multiplies two quaternion arrays
    qarray_conjugate      - returns array of conjugates of quaternion array
    qarray_mag            - returns array of magnitudes of quaternion array
    qarray_inverse        - returns array of inverse of quaternion array
    qarray_from_axangle   - returns quaternion array from array of axis angle 
                            rotations 
    qarray_from_vect      - returns quaternion array from array of vectors
    vect_from_qarray      - returns array of vectors from quaternion array
    qarray_from_euler     - returns array of quaternins from array of euler
                            angles
    euler_from_qarray     - returns array of euler angles from quaternion array
    rotate_vect_array     - rotates array of vectors by given axis angle 
                            rotations

Author: William Dickson 
--------------------------------------------------------------------------------
"""
import scipy 

# Quaterion array functions ----------------------------------------------------

def qarray_multiply(q1,q2):
    """
    Multiply two quaternion arrays

    (Q1 * Q2).w = (w1w2 - x1x2 - y1y2 - z1z2)
    (Q1 * Q2).x = (w1x2 + x1w2 + y1z2 - z1y2)
    (Q1 * Q2).y = (w1y2 - x1z2 + y1w2 + z1x2)
    (Q1 * Q2).z = (w1z2 + x1y2 - y1x2 + z1w2

    """
    qq = scipy.zeros(q1.shape)
    qq[:,0] = q1[:,0]*q2[:,0] - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2] - q1[:,3]*q2[:,3]
    qq[:,1] = q1[:,0]*q2[:,1] + q1[:,1]*q2[:,0] + q1[:,2]*q2[:,3] - q1[:,3]*q2[:,2]
    qq[:,2] = q1[:,0]*q2[:,2] - q1[:,1]*q2[:,3] + q1[:,2]*q2[:,0] + q1[:,3]*q2[:,1]
    qq[:,3] = q1[:,0]*q2[:,3] + q1[:,1]*q2[:,2] - q1[:,2]*q2[:,1] + q1[:,3]*q2[:,0]
    return qq

def qarray_conjugate(q):
    """
    Compute conjugates of quaternion array  
    """
    qc = scipy.zeros(q.shape)
    qc[:,0] = q[:,0]
    qc[:,1] = -q[:,1]
    qc[:,2] = -q[:,2]
    qc[:,3] = -q[:,3]
    return qc

def qarray_mag(q):
    """
    Return magitudes of quaternion array
    """
    mag = scipy.zeros((q.shape[0],1))
    mag[:,0] = scipy.sqrt(q[:,0]**2 + q[:,1]**2 + q[:,2]**2 + q[:,3]**2)
    return mag

def qarray_inverse(q):
    """
    Compute inverses of quaternion array
    """
    qc = qarray_conjugate(q)
    qmag = qarray_mag(q)
    qinv = qc*(1.0/qmag)
    return qinv
    
def qarray_from_axang(ax,ang):
    """
    Compute quanterion array from axis and angle arrays
    """    
    q = scipy.zeros((ax.shape[0],4))
    q[:,0] = scipy.cos(0.5*ang)
    q[:,1] = ax[:,0]*scipy.sin(0.5*ang)
    q[:,2] = ax[:,1]*scipy.sin(0.5*ang)
    q[:,3] = ax[:,2]*scipy.sin(0.5*ang)
    return q
    
def qarray_from_vect(v):
    """
    Get quaternion array from array of vectors
    """
    q = scipy.zeros((v.shape[0],4))
    q[:,1:] = scipy.array(v)
    return q

def vect_from_qarray(q):
    """
    Return vector array from quaternion array
    """
    v = scipy.array(q[:,1:])
    return v

def qarray_from_euler(euler_angs):
    """
    Get quaternion array from array of euler angles. Note, array of euler angles should 
    be in columns of [heading,attitude,bank].

    Euler angle convention - similar to NASA standard airplane, but with y and z axes 
    swapped to conform with x3d. (in order of application)
        1.0 rotation about y-axis
        2.0 rotation about z-axis
        3.0 rotation about x-axis
    """
    head = euler_angs[:,0]
    atti = euler_angs[:,1]
    bank = euler_angs[:,2]

    c0 = scipy.cos(head/2.0)
    c1 = scipy.cos(atti/2.0)
    c2 = scipy.cos(bank/2.0)
    s0 = scipy.sin(head/2.0)
    s1 = scipy.sin(atti/2.0)
    s2 = scipy.sin(bank/2.0)
    
    w = c0*c1*c2 - s0*s1*s2
    x = s0*s1*c2 + c0*c1*s2
    y = s0*c1*c2 + c0*s1*s2
    z = c0*s1*c2 - s0*c1*s2

    q = scipy.zeros((euler_angs.shape[0], 4))
    q[:,0] = w
    q[:,1] = x
    q[:,2] = y
    q[:,3] = z

    return q

def euler_from_qarray(q,tol=0.499):
    """
    Get array of euler angles from array of quaternions. Note, array of euler angles 
    will be in columns of [heading,attitude,bank].

    Euler angle convention - similar to NASA standard airplane, but with y and z axes 
    swapped to conform with x3d. (in order of application)
        1.0 rotation about y-axis
        2.0 rotation about z-axis
        3.0 rotation about x-axis
    """
    qw = q[:,0]
    qx = q[:,1]
    qy = q[:,2]
    qz = q[:,3]

    head = scipy.zeros((q.shape[0],))
    atti = scipy.zeros((q.shape[0],))
    bank = scipy.zeros((q.shape[0],))

    test = qx*qy - qz*qw # test for north of south pole

    # Points not at north or south pole
    mask0 = scipy.logical_and(test <= tol, test >= -tol)
    qw_0 = qw[mask0]
    qx_0 = qx[mask0]
    qy_0 = qy[mask0]
    qz_0 = qz[mask0]
    head[mask0] = scipy.arctan2(2.0*qy_0*qw_0 - 2.0*qx_0*qz_0, 1.0 - 2.0*qy_0**2 - 2.0*qz_0**2)
    atti[mask0] = scipy.arcsin(2.0*qx_0*qy_0 + 2.0*qz_0*qw_0)
    bank[mask0] = scipy.arctan2(2.0*qx_0*qw_0 - 2.0*qy_0*qz_0, 1.0 - 2.0*qx_0**2 - 2.0*qz_0**2)

    # Points at north pole
    mask1 = test > tol
    qw_1 = qw[mask1]
    qx_1 = qx[mask1]
    qy_1 = qy[mask1]
    qz_1 = qz[mask1]
    head[mask1] = 2.0*scipy.arctan2(qx_1,qw_1)
    atti[mask1] = scipy.arcsin(2.0*qx_1*qy_1 + 2.0*qz_1*qw_1)
    bank[mask1] = scipy.zeros((qw_1.shape[0],))

    # Points at south pole
    mask2 = test < -tol
    qw_2 = qw[mask2]
    qx_2 = qx[mask2]
    qy_2 = qy[mask2]
    qz_2 = qz[mask2]
    head[mask2] = -2.0*scipy.arctan2(qx_2,qw_2)
    atti[mask2] = scipy.arcsin(2.0*qx_2*qy_2 + 2.0*qz_2*qw_2)
    bank[mask2] = scipy.zeros((qw_2.shape[0],))

    euler_angs = scipy.zeros((q.shape[0],3))
    euler_angs[:,0] = head
    euler_angs[:,1] = atti
    euler_angs[:,2] = bank
    return euler_angs

def rotate_vect_array(v_mat, ax, ang):
    """
    Rotate a vector array by a given angle about a given axis or about a
    given array of axes and angles.

    v_mat = vector array (N,3)
    ax    = single rotation axis or an array of rotation axis.
            If ax is a single rotation axes then it must be a three element
            list or tuple or array. If ax is an array of rotation axes then
            it must have shape (N,3) array.
    ang   = single rotation angle or an array of angles in radians. If ang is
            a sigle rotation angle it must be a float or int. If ang is an
            array of rotation axes then it must have shape (N,), (N,1), or (1,N) 
    
    """
    # Convert input vector/vector-array to appropriate shape 
    if type(v_mat) in [list, tuple]:
        _v_mat = scipy.array([v_mat])
    elif v_mat.shape in [(3,), (3,1)]:
        _v_mat = scipy.reshape(v_mat,(1,3))
    else:
        _v_mat = v_mat
    # Convert axes/axes-array to apprpriate shape
    if type(ax) in [list, tuple]:
        _ax = scipy.array([ax])
    elif ax.shape in [(3,), (3,1)]:
        _ax = scipy.reshape(ax,(1,3))
    else:
        _ax = ax
    # Convert ang to appropriate type
    if type(ang) in [float, int]:
        _ang = scipy.array(float(ang))
    else:
        _ang = ang
    
    # Deal w/ shape issues
    if _ax.shape[0] == 1:
        _ax = scipy.ones(v_mat.shape)*_ax
    if _ang.shape == (1,):
        _ang = scipy.ones((v_mat.shape,1))*_ang
    
    # Rotate vector matrix using matrices of axes and angles
    qrot = qarray_from_axang(_ax, _ang)
    qrot_inv = qarray_inverse(qrot)
    qvect = qarray_from_vect(_v_mat)
    temp = qarray_multiply(qvect, qrot_inv)
    qvect_new = qarray_multiply(qrot, temp)
    v_mat_new = vect_from_qarray(qvect_new)
    return v_mat_new

def rotate_euler_array(euler_angs, ax, ang):
    """
    Rotate array of euler angles by axis angle rotations. Note, the axis angle
    rotations my be a single rotation axis and a single angle, or an array of
    rotation axes and and array of rotation angles.

    """
    N = euler_angs.shape[0]
    if ax.shape == (3,):
        _ax = scipy.ones((N,3))
        _ax[:,0] = ax[0]
        _ax[:,1] = ax[1]
        _ax[:,2] = ax[2]
        _ang = scipy.zeros((N,))
        _ang[:] = float(ang)
    elif ax.shape == (N,3):
        _ax = ax
        if not ang.shape == (N,):
            raise ValueError, 'if axis shape = (N,3) then ang shape must be (N,)'
        _ang = ang
    else:
        raise ValueError, 'axis shape must be (3,) or (N,3)'

    q_rot = qarray_from_axang(_ax, _ang)
    q_euler = qarray_from_euler(euler_angs)
    q_euler_rotated = qarray_multiply(q_rot,q_euler)
    euler_angs_new = euler_from_qarray(q_euler_rotated)
    return euler_angs_new

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # A simple example
    import pylab
    D2R = scipy.pi/180.0
    R2D = 1.0/D2R
    T = 1.0
    N = 500
    t = scipy.linspace(0,T,N)
    euler_angs = scipy.zeros((N,3))
    euler_angs[:,0] = scipy.cos(2.0*scipy.pi*t/T)
    ax = scipy.array([0.0,0.0,1.0])
    ang = -10.0*D2R
    euler_angs_rotated = rotate_euler_array(euler_angs,ax,ang)

    pylab.figure(1)
    pylab.subplot(311)
    pylab.plot(t,R2D*euler_angs[:,0])
    pylab.subplot(312)
    pylab.plot(t,R2D*euler_angs[:,1])
    pylab.subplot(313)
    pylab.plot(t,R2D*euler_angs[:,2])

    pylab.figure(2)
    pylab.subplot(311)
    pylab.plot(t,R2D*euler_angs_rotated[:,0])
    pylab.subplot(312)
    pylab.plot(t,R2D*euler_angs_rotated[:,1])
    pylab.subplot(313)
    pylab.plot(t,R2D*euler_angs_rotated[:,2])
    pylab.show()



