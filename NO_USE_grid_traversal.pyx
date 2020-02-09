"""
Simple integrators for the radiative transfer equation
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
cimport numpy as np
cimport cython
#cimport healpix_interface
from libc.stdlib cimport malloc, calloc, free, abs
from libc.math cimport exp, floor, log2, \
    fabs, atan, atan2, asin, cos, sin, sqrt, acos, M_PI, sqrt
from yt.utilities.lib.fp_utils cimport imax, fmax, imin, fmin, iclip, fclip, i64clip
from field_interpolation_tables cimport \
    FieldInterpolationTable, FIT_initialize_table, FIT_eval_transfer,\
    FIT_eval_transfer_with_light
from fixed_interpolator cimport *

DEF Nch = 4


#-----------------------------------------------------------------------------
# walk_volume(VolumeContainer *vc,  np.float64_t v_pos[3], np.float64_t v_dir[3], sampler_function *sample,
#             void *data, np.float64_t *return_t = NULL, np.float64_t max_t = 1.0)
#      vc        VolumeContainer*  : Pointer to the volume container to be traversed.
#      v_pos     np.float64_t[3]   : The x,y,z coordinates of the ray's origin.
#      v_dir     np.float64_t[3]   : The x,y,z coordinates of the ray's direction.
#      sample    sampler_function* : Pointer to the sample function to be used.
#      return_t  np.float64_t*     : # TODO: Unsure of behavior. Defaulted to NULL.
#      max_t     np.float64_t      : # TODO: Unsure of behavior. Defaulted to 1.0.
#
# Written by the yt Development Team.
# Encapsulates the Amanatides & Woo "Fast Traversal Voxel Algorithm" to walk over a volume container 'vc'
# The function occurs in two phases, initialization and traversal. 
# See: https://www.researchgate.net/publication/2611491_A_Fast_Voxel_Traversal_Algorithm_for_Ray_Tracing
# TODO: Add more. documentation. May be more readable to break this function up as well.
#-----------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int walk_volume(VolumeContainer *vc,
                     np.float64_t v_pos[3],
                     np.float64_t v_dir[3],
                     sampler_function *sample,
                     void *data,
                     np.float64_t *return_t = NULL,
                     np.float64_t max_t = 1.0) nogil:
    cdef int cur_ind[3]
    cdef int step[3]
    cdef int x, y, i, hit, direction
    cdef np.float64_t intersect_t = 1.1
    cdef np.float64_t iv_dir[3]
    cdef np.float64_t tmax[3]
    cdef np.float64_t tdelta[3]
    cdef np.float64_t exit_t = -1.0, enter_t = -1.0
    cdef np.float64_t tl, temp_x, temp_y = -1
    if max_t > 1.0: max_t = 1.0
    direction = -1
    if vc.left_edge[0] <= v_pos[0] and v_pos[0] < vc.right_edge[0] and \
       vc.left_edge[1] <= v_pos[1] and v_pos[1] < vc.right_edge[1] and \
       vc.left_edge[2] <= v_pos[2] and v_pos[2] < vc.right_edge[2]:
        intersect_t = 0.0
        direction = 3
    for i in range(3):
        if (v_dir[i] < 0):
            step[i] = -1
        elif (v_dir[i] == 0.0):
            step[i] = 0
            continue
        else:
            step[i] = 1
        iv_dir[i] = 1.0/v_dir[i]
        if direction == 3: continue
        x = (i+1) % 3
        y = (i+2) % 3
        if step[i] > 0:
            tl = (vc.left_edge[i] - v_pos[i])*iv_dir[i]
        else:
            tl = (vc.right_edge[i] - v_pos[i])*iv_dir[i]
        temp_x = (v_pos[x] + tl*v_dir[x])
        temp_y = (v_pos[y] + tl*v_dir[y])
        if fabs(temp_x - vc.left_edge[x]) < 1e-10*vc.dds[x]:
            temp_x = vc.left_edge[x]
        elif fabs(temp_x - vc.right_edge[x]) < 1e-10*vc.dds[x]:
            temp_x = vc.right_edge[x]
        if fabs(temp_y - vc.left_edge[y]) < 1e-10*vc.dds[y]:
            temp_y = vc.left_edge[y]
        elif fabs(temp_y - vc.right_edge[y]) < 1e-10*vc.dds[y]:
            temp_y = vc.right_edge[y]
        if vc.left_edge[x] <= temp_x and temp_x <= vc.right_edge[x] and \
           vc.left_edge[y] <= temp_y and temp_y <= vc.right_edge[y] and \
           0.0 <= tl and tl < intersect_t:
            direction = i
            intersect_t = tl
    if enter_t >= 0.0: intersect_t = enter_t 
    if not ((0.0 <= intersect_t) and (intersect_t < max_t)): return 0
    for i in range(3):
        # Two things have to be set inside this loop.
        # cur_ind[i], the current index of the grid cell the ray is in
        # tmax[i], the 't' until it crosses out of the grid cell
        tdelta[i] = step[i] * iv_dir[i] * vc.dds[i]
        if i == direction and step[i] > 0:
            # Intersection with the left face in this direction
            cur_ind[i] = 0
        elif i == direction and step[i] < 0:
            # Intersection with the right face in this direction
            cur_ind[i] = vc.dims[i] - 1
        else:
            # We are somewhere in the middle of the face
            temp_x = intersect_t * v_dir[i] + v_pos[i] # current position
            temp_y = ((temp_x - vc.left_edge[i])*vc.idds[i])
            # There are some really tough cases where we just within a couple
            # least significant places of the edge, and this helps prevent
            # killing the calculation through a segfault in those cases.
            if -1 < temp_y < 0 and step[i] > 0:
                temp_y = 0.0
            elif vc.dims[i] - 1 < temp_y < vc.dims[i] and step[i] < 0:
                temp_y = vc.dims[i] - 1
            cur_ind[i] =  <int> (floor(temp_y))
        if step[i] > 0:
            temp_y = (cur_ind[i] + 1) * vc.dds[i] + vc.left_edge[i]
        elif step[i] < 0:
            temp_y = cur_ind[i] * vc.dds[i] + vc.left_edge[i]
        tmax[i] = (temp_y - v_pos[i]) * iv_dir[i]
        if step[i] == 0:
            tmax[i] = 1e60
    # We have to jumpstart our calculation
    for i in range(3):
        if cur_ind[i] == vc.dims[i] and step[i] >= 0:
            return 0
        if cur_ind[i] == -1 and step[i] <= -1:
            return 0
    enter_t = intersect_t
    hit = 0
    while 1:
        hit += 1
        if tmax[0] < tmax[1]:
            if tmax[0] < tmax[2]:
                i = 0
            else:
                i = 2
        else:
            if tmax[1] < tmax[2]:
                i = 1
            else:
                i = 2
        exit_t = fmin(tmax[i], max_t)
        sample(vc, v_pos, v_dir, enter_t, exit_t, cur_ind, data)
        cur_ind[i] += step[i]
        enter_t = tmax[i]
        tmax[i] += tdelta[i]
        if cur_ind[i] < 0 or cur_ind[i] >= vc.dims[i] or enter_t >= max_t:
            break
    if return_t != NULL: return_t[0] = exit_t
    return hit
