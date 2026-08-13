# ruff: noqa: N803, N806
"""Defines utility functions for the rolland project.

autosummary::
    :toctree: func

    build_equ_sleeper_matrix
    build_fnd_damp_matrix
    build_fnd_stiff_matrix
    build_pad_ballast_damp_matrices
    build_pad_ballast_stiff_matrices
    build_rail_matrices
    build_sleep_mass_matrix
    build_transfm_matrices
    calc_cut_on_frequ
"""

from numpy import argmax, array, block, diag, ones, pi, sqrt, zeros
from scipy.linalg import eigh


def build_rail_matrices(rail, damp_type="hysteretic"):
    """Build rail matrices.

    Attributes
    ----------
    rail : Rail
        The rail object.
    damp_type : str, default="hysteretic"
        The Type of the used damping Model for components. "viscous" uses the viscous damping model
        and "hysteretic" uses the hysteretic damping model.

    Return
    ----------
    K0 : ndarray
        Rail stiffness matrix.
    K1 : ndarray
        Rail stiffness matrix.
    K2 : ndarray
        Rail stiffness matrix.
    Mr : ndarray
        Rail mass matrix.
    """
    # Rail properties
    G = rail.G
    E = rail.E if damp_type == "viscous" else rail.E * (1 + 1j * rail.etar)
    A = rail.Ar
    Iy = rail.Iyr
    Iz = rail.Izr
    Iwz = rail.Iwz
    Iwy = rail.Iwy
    Iyz = rail.Iyz
    rho = rail.rho
    mr = rail.mr
    kap_y = rail.kapy
    kap_z = rail.kapz
    e_y = rail.ey
    e_z = rail.ez
    Iw = rail.Iw
    J = rail.J
    Jt = rail.J_t
    Ip = rail.Ipr

    ## Matrix construction
    neg = 1
    # K0
    K0 = array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, G * A * kap_y, 0, G * A * kap_y * e_z * neg],
            [0, 0, 0, 0, 0, G * A * kap_z, G * A * kap_z * e_y * neg],
            [0, 0, 0, 0, G * A * kap_y * e_z, G * A * kap_z * e_y, G * Jt],
        ],
    )

    # K1
    K1 = array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, G * A * kap_z, G * A * kap_z * e_y * neg],
            [0, 0, 0, 0, -G * A * kap_y, 0, -G * A * kap_y * e_z * neg],
            [0, 0, 0, 0, -G * A * kap_y * e_z, -G * A * kap_z * e_y, -G * Jt],
            [0, 0, G * A * kap_y, G * A * kap_y * e_z, 0, 0, 0],
            [0, -G * A * kap_z, 0, G * A * kap_z * e_y, 0, 0, 0],
            [0, -G * A * kap_z * e_y, G * A * kap_y * e_z, G * Jt, 0, 0, 0],
        ],
    )

    # K2
    K2 = array(
        [
            [-E * A, 0, 0, 0, 0, 0, 0],
            [0, -G * A * kap_z, 0, G * A * kap_z * e_y, 0, 0, 0],
            [0, 0, -G * A * kap_y, -G * A * kap_y * e_z, 0, 0, 0],
            [0, G * A * kap_z * e_y, -G * A * kap_y * e_z, -G * (Jt + J), 0, 0, 0],
            [0, 0, 0, 0, -E * Iz, E * Iyz, -E * Iwz * neg],
            [0, 0, 0, 0, E * Iyz, -E * Iy, E * Iwy * neg],
            [0, 0, 0, 0, -E * Iwz, E * Iwy, -E * Iw],
        ],
    )

    # Mr
    Mr = array(
        [
            [mr, 0, 0, 0, 0, 0, 0],
            [0, mr, 0, 0, 0, 0, 0],
            [0, 0, mr, 0, 0, 0, 0],
            [0, 0, 0, rho * Ip, 0, 0, 0],
            [0, 0, 0, 0, rho * (Iz - Iwz), -rho * Iyz, rho * Iwz * neg],
            [0, 0, 0, 0, -rho * Iyz, rho * (Iy + Iwy), -rho * Iwy * neg],
            [0, 0, 0, 0, rho * Iwz, -rho * Iwy, rho * Iw],
        ],
    )
    return K0, K1, K2, Mr



def build_transfm_matrices(z_f, y_f, z_st=0, z_sb=0, chi_f=0):
    """Build transformation matrices according to :cite:p:`kostovasilis_semi-analytical_2017`.

    Attributes
    ----------
    z_f : float
        The vertical distance from the rail foot to the rail centroid :math:`[m]`.
    y_f : float
        The lateral distance from the rail foot to the rail centroid :math:`[m]`.
    z_st : float, default=0
        The vertical distance from the sleeper centroid to the sleeper top surface :math:`[m]`.
    z_sb : float, default=0
        The vertical distance from the sleeper bottom surface to the sleeper centroid :math:`[m]`.
    chi_f : float, default=0
        The warping function of the cross-section at the rail foot :math:`[m^2]`.

    Return
    ----------
    Tf: ndarray
        The transformation matrix from the rail centroid to the rail foot.
    Tst: ndarray
        The transformation matrix from the sleeper centroid to the rail foot.
    Tsb: ndarray
        The transformation matrix from the sleeper centroid to the sleeper bottom.
    """
    # Foot transform
    Tf = array(
        [
            [1, 0, 0, 0, y_f, -z_f, chi_f],
            [0, 1, 0, -y_f, 0, 0, 0],
            [0, 0, 1, z_f, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
    )

    # T_st --> matrix for the transformation  of the co-ordinates from the six degrees of freedom
    # at the sleeper centroid to the seven  degrees of freedom at the sleeper top surface.
    Tst = array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, z_st, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )

    # T_sb --> matrix for the transformation of the coordinates from
    # the bottom  surface of the sleeper to the sleeper centroid,
    Tsb = array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, z_sb, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )
    return Tf, Tst, Tsb



def build_equ_sleeper_matrix(track, seclay=None):
    """Build equivalent sleeper matrix based on :cite:p:`kostovasilis_analytical_2017`.

    Attributes
    ----------
    track : Track
        The track object.
    seclay : Sleeper/Slab, optional
        The specific secondary layer instance.

    Return
    ----------
    E : ndarray
        The equivalent sleeper Matrix, which is used to scale the stiffness of the balast.
    """
    if seclay is None:
        if hasattr(track, 'slab'):
            seclay = track.slab
            track.calc_equiv_slab_factors(slab=seclay)
        elif hasattr(track, 'sleeper'):
            seclay = track.sleeper
            track.calc_equiv_sleeper_factors(sleeper=seclay)

    f_x = seclay.f_x
    f_z = seclay.f_z

    E = ones(7)
    E[0] = 1 / f_x
    E[1] = 1 / f_z

    return E



def build_pad_ballast_stiff_matrices(track, damp_type="hysteretic", E=None, pad=None, ballast=None):
    """Build pad and ballast stiffness matrices.

    Attributes
    ----------
    track : Track
        The track object.
    damp_type : str, default="hysteretic"
        The Type of the used damping Model for components. "viscous" uses the viscous damping model
        and "hysteretic" uses the hysteretic damping model.
    E : ndarray, default=ones(7)
        The equivalent sleeper Matrix, which is used to scale the stiffness of the balast.
        If "None", a default value of ones(7) is used.
    pad : Pad, optional
        The specific pad instance.
    ballast : Ballast, optional
        The specific ballast instance.
    seclay : Sleeper/Slab, optional
        The specific secondary layer instance.

    Return
    ----------
    Kp: ndarray
        The railpad stiffness matrix.
    Kb: ndarray
        The ballast stiffness matrix.
    """
    if E is None:
        E = ones(7)

    p = pad if pad is not None else track.pad
    if pad is None:
        track.calc_pad_warping_stiffn(pad=p)  # Calculate warping stiffness if pad was not explicitly provided
    Kp = diag(
        [
            p.sp_x,
            p.sp_z,
            p.sp_y,
            p.sp_xr,
            p.sp_zr,
            p.sp_yr,
            p.sp_w,
        ],
    )

    if damp_type == "hysteretic":
        eta_p = diag(
            [
                p.etap_x,
                p.etap_z,
                p.etap_y,
                p.etap_r,
                0,
                0,
                0,
            ],
        )
        Kp = Kp * (1 + 1j * eta_p)

    else:
        pass

    b = ballast if ballast is not None else getattr(track, 'ballast', None)
    if b is not None:
        if ballast is None and (hasattr(track, 'sleeper') or hasattr(track, 'slab')):
            # Only recalculate rotational stiffness if ballast was not explicitly provided
            track.calc_ballast_rotational_stiffn(ballast=b)

        Kb = diag(
            [
                b.sb_x,
                b.sb_z,
                b.sb_y,
                b.sb_xr,
                b.sb_zr,
                b.sb_yr,
                0,
            ],
        )

        if damp_type == "hysteretic":
            eta_b = diag(
                [
                    b.etab_x,
                    b.etab_z,
                    b.etab_y,
                    b.etab_r,
                    b.etab_r,
                    b.etab_r,
                    1e-20,
                ],
            )
            Kb = Kb * (1 + 1j * eta_b)
        else:
            pass

    else:
        Kb = zeros((7, 7))

    return Kp, Kb * E



def build_sleep_mass_matrix(track, E=None, seclay=None):
    """Build sleeper mass matrix.

    Attributes
    ----------
    track : Track
        The track object.
    E : ndarray, default=ones(7)
        The equivalent sleeper Matrix, which is used to scale the mass of the balast.
        If "None", a default value of ones(7) is used.
    seclay : Sleeper/Slab, optional
        The specific secondary layer instance.

    Return
    ----------
    Ms : ndarray
        Sleeper mass matrix.
    """
    if E is None:
        E = ones(7)

    if seclay is None:
        seclay = track.slab if hasattr(track, 'slab') else track.sleeper

    Ms = diag(
        [
            seclay.ms,
            seclay.ms,
            seclay.ms,
            seclay.rhos * seclay.Is_x,
            seclay.rhos * seclay.Is_z,
            seclay.rhos * seclay.Is_y,
            1e-20,
        ],
    )

    return Ms * E



def build_fnd_stiff_matrix(Kp, Tf, Kb=None, Tst=None, Tsb=None):
    """Build foundation stiffness matrix.

    Attributes
    ----------
    Kp: ndarray
        The railpad stiffness matrix.
    Tf: ndarray
        The transformation matrix from the rail centroid to the rail foot.
    Kb: ndarray, default=zeros((7, 7))
        The ballast stiffness matrix.
    Tst: ndarray, default=zeros((7, 7))
        The transformation matrix from the sleeper centroid to the rail foot.
    Tsb: ndarray, default=zeros((7, 7))
        The transformation matrix from the sleeper centroid to the sleeper bottom.

    Return
    ----------
    K_fnd : ndarray
        Foundation stiffness matrix.
    """
    if Kb is None:
        Kb = zeros((7, 7))
    if Tst is None:
        Tst = zeros((7, 7))
    if Tsb is None:
        Tsb = zeros((7, 7))

    Kp_rf_rc = Tf.T @ Kp @ Tf  # TM rail foot --> rail centroid
    Kp_sc_rc = -Tf.T @ Kp @ Tst  # TM sleeper centroid --> rail centroid
    Kp_rc_sc = -Tst.T @ Kp @ Tf  # TM rail centroid --> sleeper centroid
    Kp_sc_rf = Tst.T @ Kp @ Tst  # TM sleeper centroid --> rail foot
    Ks = Tsb.T @ Kb @ Tsb  # TM sleeper centroid --> ground
    return block([[Kp_rf_rc, Kp_sc_rc], [Kp_rc_sc, Kp_sc_rf + Ks]])

def build_fnd_damp_matrix(Dp, Tf, Db=None, Tst=None, Tsb=None):
    """Build foundation damping matrix.

    Attributes
    ----------
    Dp : ndarray
        The Railpad damping matrix.
    Tf : ndarray
        The transformation matrix from rail foot to rail centroid.
    Db : ndarray, default=zeros((7, 7))
        The ballast damping matrix.
    Tst : ndarray, default=zeros((7, 7))
        The transformation matrix from the sleeper centroid to the rail foot.
    Tsb : ndarray, default=zeros((7, 7))
        The transformation matrix from the sleeper centroid to the sleeper bottom.

    Return
    ----------
    D_fnd : ndarray
        Foundation damping matrix.
    """
    if Db is None:
        Db = zeros((7, 7))
    if Tst is None:
        Tst = zeros((7, 7))
    if Tsb is None:
        Tsb = zeros((7, 7))

    Dp_rf_rc = Tf.T @ Dp @ Tf  # TM rail foot --> rail centroid
    Dp_sc_rc = -Tf.T @ Dp @ Tst  # TM sleeper centroid --> rail centroid
    Dp_rc_sc = -Tst.T @ Dp @ Tf  # TM rail centroid --> sleeper centroid
    Dp_sc_rf = Tst.T @ Dp @ Tst  # TM sleeper centroid --> rail foot
    Ds = Tsb.T @ Db @ Tsb  # TM sleeper centroid --> ground
    return block([[Dp_rf_rc, Dp_sc_rc], [Dp_rc_sc, Dp_sc_rf + Ds]])

def calc_cut_on_frequ(K0, K_fnd, Mr, Ms=None):
    """Calculate the cut-on frequencies of the track system.

    Attributes
    ----------
    K0 : ndarray
        Rail stiffness matrix.
    K_fnd : ndarray
        Foundation stiffness matrix.
    Mr : ndarray
        Rail mass matrix.
    Ms : ndarray, default=diag(ones(7) * 1e-20)
        Sleeper mass matrix.

    Return
    ----------
    cof : ndarray
        Cuton frequencies of the track system.
    """
    if Ms is None:
        Ms = diag(ones(7) * 1e-20)
    M = block([[Mr, zeros((7, 7))], [zeros((7, 7)), Ms]])
    K = K_fnd + block([[K0, zeros((7, 7))], [zeros((7, 14))]])

    eigvals, eigvecs = eigh(K, M)
    cof = zeros(M.shape[0])

    eigfrqu = sqrt(abs(eigvals)) / (2 * pi)
    energy_contrib = abs(eigvecs * (M @ eigvecs))

    for i in range(len(cof)):
        dof_energies = energy_contrib[i, :]
        best_mode_idx = argmax(dof_energies)
        cof[i] = eigfrqu[best_mode_idx]

    return cof



def build_pad_ballast_damp_matrices(track, cof, E=None, pad=None, ballast=None):
    """Build pad and ballast damping matrices.

    Attributes
    ----------
    track : Track
        The track object.
    cof : ndarray
        The cut-on frequencies.
    E : ndarray, default=ones(7)
        The equivalent sleeper Matrix, which is used to scale the mass of the balast.
        If "None", a default value of ones(7) is used.
    pad : Pad, optional
        The specific pad instance.
    ballast : Ballast, optional
        The specific ballast instance.

    Return
    ----------
    Dp : ndarray
        The railpad damping matrix.
    Db: ndarray
        The ballast damping matrix.
    """
    if E is None:
        E = ones(7)

    p = pad if pad is not None else track.pad
    if pad is None:
        track.calc_pad_viscous_damp_cuton(pad=p, cof=cof)

    Dp = diag(
        [
            p.dp_x,
            p.dp_z,
            p.dp_y,
            p.dp_xr,
            0,
            0,
            0,
        ],
    )

    b = ballast if ballast is not None else getattr(track, 'ballast', None)
    if b is not None:
        if ballast is None:
            track.calc_ballast_viscous_damp_cuton(ballast=b, cof=cof)

        Db = diag(
            [
                b.db_x,
                b.db_z,
                b.db_y,
                b.db_xr,
                b.db_zr,
                b.db_yr,
                0,
            ],
        )

    else:
        Db = zeros((7, 7))

    return Dp, Db * E
