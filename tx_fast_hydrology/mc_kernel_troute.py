"""Numba port of the t-route Muskingum-Cunge routing kernel."""
import numpy as np
from numba import njit


@njit(cache=True)
def _hydraulic_geometry(h, bfd, bw, twcc, z):
    """
    t-route hydraulic_geometry: returns (twl, R, AREA, AREAC, WP, WPC,
    h_lt_bf, h_gt_bf)
    """
    twl = bw + 2.0*z*h

    # Split depth between the main channel and compound section.
    h_gt_bf = max(h - bfd, 0.0)
    h_lt_bf = min(bfd, h)

    # With no compound-channel width, extend the trapezoidal main channel.
    if (h_gt_bf > 0.0) and (twcc <= 0.0):
        h_gt_bf = 0.0
        h_lt_bf = h

    AREA = (bw + h_lt_bf*z) * h_lt_bf
    WP = bw + 2.0*h_lt_bf*np.sqrt(1.0 + z*z)
    AREAC = twcc*h_gt_bf
    if h_gt_bf > 0.0:
        WPC = twcc + 2.0*h_gt_bf
    else:
        WPC = 0.0

    denom = WP + WPC
    if denom > 0.0:
        R = (AREA + AREAC)/denom
    else:
        R = 0.0

    return twl, R, AREA, AREAC, WP, WPC, h_lt_bf, h_gt_bf


@njit(cache=True)
def _secant2_h(z, bw, bfd, twcc, s0, n, ncc, dt, dx,
               qdp, ql, qup, quc, h, interval,
               Qj_prev, C1_in, C2_in, C3_in, C4_in):
    """
    t-route secant2_h - `interval` corresponds to 1 (upper) or 2 (lower).

    Returns (Qj, C1, C2, C3, C4, K, X).
    """
    twl, R, AREA, AREAC, WP, WPC, h_lt_bf, h_gt_bf = \
        _hydraulic_geometry(h, bfd, bw, twcc, z)

    # kinematic celerity Ck
    if (h > bfd) and (twcc > 0.0) and (ncc > 0.0):
        Ck = max(0.0, ((np.sqrt(s0)/n)
              * ((5.0/3.0)*R**(2.0/3.0)
                 - ((2.0/3.0)*R**(5.0/3.0)
                    * (2.0*np.sqrt(1.0 + z*z)/(bw + 2.0*bfd*z))))
              * AREA
              + ((np.sqrt(s0)/ncc)*(5.0/3.0)*(h - bfd)**(2.0/3.0))*AREAC)
              /(AREA + AREAC))
    else:
        if h > 0.0:
            Ck = max(0.0, (np.sqrt(s0)/n)
                   * ((5.0/3.0)*R**(2.0/3.0)
                      - ((2.0/3.0)*R**(5.0/3.0)
                         * (2.0*np.sqrt(1.0 + z*z)/(bw + 2.0*h*z)))))
        else:
            Ck = 0.0

    # K
    if Ck > 0.0:
        Km = max(dt, dx/Ck)
    else:
        Km = dt

    # X
    X = 0.0
    if (h > bfd) and (twcc > 0.0) and (ncc > 0.0) and (Ck > 0.0):
        if interval == 1:
            X = min(0.5, max(0.0, 0.5 * (1.0 - (Qj_prev / (2.0 * twcc * s0 * Ck * dx)))))
        else:
            base = (C1_in * qup) + (C2_in * quc) + (C3_in * qdp) + C4_in
            X = min(0.5, max(0.25, 0.5 * (1.0 - (base / (2.0 * twcc * s0 * Ck * dx)))))
    else:
        if Ck > 0.0:
            if interval == 1:
                X = min(0.5, max(0.0, 0.5 * (1.0 - (Qj_prev / (2.0 * twl * s0 * Ck * dx)))))
            else:
                base = (C1_in * qup) + (C2_in * quc) + (C3_in * qdp) + C4_in
                X = min(0.5, max(0.25, 0.5 * (1.0 - (base / (2.0 * twl * s0 * Ck * dx)))))
        else:
            X = 0.5

    # The rest
    D = Km * (1.0 - X) + dt / 2.0
    if D == 0.0:
        D = 1e-8

    C1 = (Km * X + dt / 2.0) / D
    C2 = (dt / 2.0 - Km * X) / D
    C3 = (Km * (1.0 - X) - dt / 2.0) / D
    C4 = (ql * dt) / D

    if interval == 2:
        if (C4 < 0.0) and (abs(C4) > (C1 * qup) + (C2 * quc) + (C3 * qdp)):
            C4 = -((C1 * qup) + (C2 * quc) + (C3 * qdp))

    if (WP + WPC) > 0.0:
        Qj = ((C1 * qup) + (C2 * quc) + (C3 * qdp) + C4) \
            - ((1.0 / (((WP * n) + (WPC * ncc)) / (WP + WPC)))
               * (AREA + AREAC) * (R ** (2.0 / 3.0)) * np.sqrt(s0))
    else:
        Qj = 0.0

    return Qj, C1, C2, C3, C4, Km, X


@njit(cache=True)
def submuskingcunge(qup, quc, qdp, ql, dt, s0, dx, n, cs, bw, tw, twcc, ncc,
                    depthp):
    """
    Single-reach solve.

    Returns (qdc, velc, depthc, K, X).
    """
    maxiter = 100
    mindepth = 0.01
    tries = 0

    C1 = 0.0
    C2 = 0.0
    C3 = 0.0
    C4 = 0.0
    # These are also the dry-reach values produced by _secant2_h when the
    # kinematic celerity is zero. Initializing them here makes the hydraulic
    # parameters explicit even when the secant solve is skipped.
    Km = dt
    X = 0.5

    if cs == 0.0:
        z = 1.0
    else:
        z = 1.0 / cs

    if bw > tw:
        bfd = bw / 0.00001
    elif bw == tw:
        bfd = bw / (2.0 * z)
    else:
        bfd = (tw - bw) / (2.0 * z)

    depthc = max(depthp, 0.0)
    h = (depthc * 1.33) + mindepth
    h_0 = (depthc * 0.67)

    if (ql > 0.0) or (qup > 0.0) or (quc > 0.0) or (qdp > 0.0):

        while True:  # goto 110
            iter_ = 0
            rerror = 1.0
            aerror = 0.01
            Qj_0 = 0.0

            while (rerror > 0.01) and (aerror >= mindepth) and (iter_ <= maxiter):
                # upper interval (h_0): X uses running Qj_0
                Qj_0, C1, C2, C3, C4, Km, X = _secant2_h(
                    z, bw, bfd, twcc, s0, n, ncc, dt, dx,
                    qdp, ql, qup, quc, h_0, 1,
                    Qj_0, C1, C2, C3, C4)
                # lower interval (h): X uses coeffs just produced by upper call
                Qj, C1, C2, C3, C4, Km, X = _secant2_h(
                    z, bw, bfd, twcc, s0, n, ncc, dt, dx,
                    qdp, ql, qup, quc, h, 2,
                    Qj_0, C1, C2, C3, C4)

                if (Qj_0 - Qj) != 0.0:
                    h_1 = h - ((Qj * (h_0 - h)) / (Qj_0 - Qj))
                    if h_1 < 0.0:
                        h_1 = h
                else:
                    h_1 = h

                if h > 0.0:
                    rerror = abs((h_1 - h) / h)
                    aerror = abs(h_1 - h)
                else:
                    rerror = 0.0
                    aerror = 0.9

                h_0 = max(0.0, h)
                h = max(0.0, h_1)
                iter_ += 1

                if h < mindepth:
                    break

            if iter_ >= maxiter and tries <= 4:
                tries += 1
                h = h * 1.33
                h_0 = h_0 * 0.67
                maxiter += 25
                continue
            break

        # final flow update
        if ((C1 * qup) + (C2 * quc) + (C3 * qdp) + C4) < 0.0:
            if (C4 < 0.0) and (abs(C4) > (C1 * qup) + (C2 * quc) + (C3 * qdp)):
                qdc = 0.0
            else:
                qdc = max(((C1 * qup) + (C2 * quc) + C4),
                          ((C1 * qup) + (C3 * qdp) + C4))
        else:
            qdc = ((C1 * qup) + (C2 * quc) + (C3 * qdp) + C4)

        twl = bw + 2.0 * z * h
        R = (h * (bw + twl) / 2.0) / (bw + 2.0 * (((twl - bw) / 2.0) ** 2.0 + h ** 2) ** 0.5)
        velc = (1.0 / n) * (R ** (2.0 / 3.0)) * np.sqrt(s0)
        depthc = h
    else:
        qdc = 0.0
        velc = 0.0
        depthc = 0.0

    return qdc, velc, depthc, Km, X


@njit(cache=True)
def _mc_ax_bu(startnodes, endnodes, indegree, qdp, ql, depthp,
              dt, So, dx, mann_n, Cs, Bw, Tw, TwCC, nCC,
              previous_nudge, assume_short_ts=True):
    """Route one network substep using t-route's MC ordering.

    With ``assume_short_ts=False``, reaches are solved upstream-to-downstream.
    The previous upstream flow (``qup``) comes from ``qdp`` while current
    upstream flow (``quc``) is accumulated from reaches already solved during
    this substep.

    ``previous_nudge`` implements WRF-Hydro Technical Description equation
    4.7. At a gaged reach, its previous nudge is added to both upstream-flow
    terms. The downstream previous-flow term already contains that nudge
    because the nudging callback stores corrected discharge in model state.

    The default ``assume_short_ts=True`` approximation instead sets current
    upstream flow equal to previous upstream flow. This reproduces NWM output
    generated with t-route's short-timestep option.
    """
    n_reaches = endnodes.size

    qdc = np.zeros(n_reaches, dtype=np.float64)
    velc = np.zeros(n_reaches, dtype=np.float64)
    depthc = np.zeros(n_reaches, dtype=np.float64)
    Kc = np.zeros(n_reaches, dtype=np.float64)
    Xc = np.zeros(n_reaches, dtype=np.float64)

    # Previous-timestep inflow is fixed throughout the substep.
    qup_acc = np.zeros(n_reaches, dtype=np.float64)
    for upstream in range(n_reaches):
        downstream = endnodes[upstream]
        if upstream != downstream:
            qup_acc[downstream] += qdp[upstream]

    if assume_short_ts:
        # Newly calculated upstream flow cannot enter a downstream reach until
        # the following substep.
        quc_acc = qup_acc.copy()
        for reach in range(n_reaches):
            qdc_i, velc_i, depthc_i, Kc_i, Xc_i = submuskingcunge(
                qup_acc[reach] + previous_nudge[reach],
                quc_acc[reach] + previous_nudge[reach],
                qdp[reach], ql[reach], dt,
                So[reach], dx[reach], mann_n[reach], Cs[reach],
                Bw[reach], Tw[reach], TwCC[reach], nCC[reach],
                depthp[reach]
            )
            qdc[reach] = qdc_i
            velc[reach] = velc_i
            depthc[reach] = depthc_i
            Kc[reach] = Kc_i
            Xc[reach] = Xc_i
        return qdc, velc, depthc, qup_acc, quc_acc, Kc, Xc

    # Default t-route behavior when the short-timestep approximation is
    # disabled: current flow is immediately available downstream. Independent
    # headwater paths can be entered in any order; a confluence is released
    # only after all its direct upstream reaches have been solved.
    quc_acc = np.zeros(n_reaches, dtype=np.float64)
    indegree_t = indegree.copy()
    solved = 0
    for k in range(startnodes.size):
        reach = startnodes[k]
        while indegree_t[reach] == 0:
            qdc_i, velc_i, depthc_i, Kc_i, Xc_i = submuskingcunge(
                qup_acc[reach] + previous_nudge[reach],
                quc_acc[reach] + previous_nudge[reach],
                qdp[reach], ql[reach], dt,
                So[reach], dx[reach], mann_n[reach], Cs[reach],
                Bw[reach], Tw[reach], TwCC[reach], nCC[reach],
                depthp[reach]
            )
            qdc[reach] = qdc_i
            velc[reach] = velc_i
            depthc[reach] = depthc_i
            Kc[reach] = Kc_i
            Xc[reach] = Xc_i
            solved += 1

            downstream = endnodes[reach]
            if reach == downstream:
                break
            quc_acc[downstream] += qdc_i
            indegree_t[downstream] -= 1
            reach = downstream

    if solved != n_reaches:
        raise ValueError('MC topology is cyclic or has unreachable reaches')
    return qdc, velc, depthc, qup_acc, quc_acc, Kc, Xc
