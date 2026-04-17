#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Minimal 3D discontinuous Galerkin-style (P0) solver for compressible Navier-Stokes
 * on a curvilinear Cartesian mesh.
 *
 * Notes:
 * - P0 DG is equivalent to a cell-centered finite-volume update.
 * - The mesh is logically Cartesian in (xi,eta,zeta) and smoothly warped in (x,y,z).
 * - This is intentionally compact/educational rather than production-ready.
 */

enum { NVAR = 5 }; /* rho, rho*u, rho*v, rho*w, rho*E */

typedef struct {
    int nx, ny, nz, ncell;
    double gamma;
    double mu;      /* dynamic viscosity */
    double pr;      /* Prandtl */
    double c_ip;    /* interior penalty coefficient */
    double cfl;
    double t_final;
    double warp;    /* warp amplitude for curvilinear map */

    double *xc, *yc, *zc;      /* cell centers */
    double *vol;               /* cell volumes */
    double *dx, *dy, *dz;      /* local metric lengths */

    double *u;                 /* conserved vars, size ncell*NVAR */
    double *rhs;               /* residual */

    /* primitive gradients for viscous term (du_i/dx_j and dT/dx_j) */
    double *gu;                /* size ncell*3 */
    double *gv;                /* size ncell*3 */
    double *gw;                /* size ncell*3 */
    double *gT;                /* size ncell*3 */
} Solver;

static inline int idx3(const Solver *s, int i, int j, int k) {
    return (k * s->ny + j) * s->nx + i;
}

static inline double clamp_min(double x, double xmin) {
    return x < xmin ? xmin : x;
}

static void map_curvilinear(double xi, double eta, double zeta, double warp,
                            double *x, double *y, double *z) {
    const double sx = sin(M_PI * xi);
    const double sy = sin(M_PI * eta);
    const double sz = sin(M_PI * zeta);
    const double cx = cos(M_PI * xi);
    const double cy = cos(M_PI * eta);
    const double cz = cos(M_PI * zeta);

    *x = xi + warp * sx * sy * sz;
    *y = eta + 0.5 * warp * cx * sy * sz;
    *z = zeta + 0.5 * warp * sx * cy * sz;

    (void)cz;
}

static void alloc_solver(Solver *s) {
    const size_t n = (size_t)s->ncell;
    s->xc = (double *)calloc(n, sizeof(double));
    s->yc = (double *)calloc(n, sizeof(double));
    s->zc = (double *)calloc(n, sizeof(double));
    s->vol = (double *)calloc(n, sizeof(double));
    s->dx = (double *)calloc(n, sizeof(double));
    s->dy = (double *)calloc(n, sizeof(double));
    s->dz = (double *)calloc(n, sizeof(double));
    s->u = (double *)calloc(n * NVAR, sizeof(double));
    s->rhs = (double *)calloc(n * NVAR, sizeof(double));
    s->gu = (double *)calloc(n * 3, sizeof(double));
    s->gv = (double *)calloc(n * 3, sizeof(double));
    s->gw = (double *)calloc(n * 3, sizeof(double));
    s->gT = (double *)calloc(n * 3, sizeof(double));

    if (!s->xc || !s->yc || !s->zc || !s->vol || !s->dx || !s->dy || !s->dz ||
        !s->u || !s->rhs || !s->gu || !s->gv || !s->gw || !s->gT) {
        fprintf(stderr, "Allocation failure\n");
        exit(1);
    }
}

static void free_solver(Solver *s) {
    free(s->xc); free(s->yc); free(s->zc);
    free(s->vol); free(s->dx); free(s->dy); free(s->dz);
    free(s->u); free(s->rhs);
    free(s->gu); free(s->gv); free(s->gw); free(s->gT);
}

static void build_mesh(Solver *s) {
    const double dxi = 1.0 / s->nx;
    const double deta = 1.0 / s->ny;
    const double dzeta = 1.0 / s->nz;

    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx; ++i) {
                const int c = idx3(s, i, j, k);

                const double xi = (i + 0.5) * dxi;
                const double eta = (j + 0.5) * deta;
                const double zeta = (k + 0.5) * dzeta;
                map_curvilinear(xi, eta, zeta, s->warp, &s->xc[c], &s->yc[c], &s->zc[c]);

                /* approximate local metric lengths from mapped neighboring points */
                double x_p, y_p, z_p, x_m, y_m, z_m;

                map_curvilinear(fmin(1.0, xi + 0.5 * dxi), eta, zeta, s->warp, &x_p, &y_p, &z_p);
                map_curvilinear(fmax(0.0, xi - 0.5 * dxi), eta, zeta, s->warp, &x_m, &y_m, &z_m);
                s->dx[c] = hypot(hypot(x_p - x_m, y_p - y_m), z_p - z_m);

                map_curvilinear(xi, fmin(1.0, eta + 0.5 * deta), zeta, s->warp, &x_p, &y_p, &z_p);
                map_curvilinear(xi, fmax(0.0, eta - 0.5 * deta), zeta, s->warp, &x_m, &y_m, &z_m);
                s->dy[c] = hypot(hypot(x_p - x_m, y_p - y_m), z_p - z_m);

                map_curvilinear(xi, eta, fmin(1.0, zeta + 0.5 * dzeta), s->warp, &x_p, &y_p, &z_p);
                map_curvilinear(xi, eta, fmax(0.0, zeta - 0.5 * dzeta), s->warp, &x_m, &y_m, &z_m);
                s->dz[c] = hypot(hypot(x_p - x_m, y_p - y_m), z_p - z_m);

                s->dx[c] = clamp_min(s->dx[c], 1e-8);
                s->dy[c] = clamp_min(s->dy[c], 1e-8);
                s->dz[c] = clamp_min(s->dz[c], 1e-8);
                s->vol[c] = s->dx[c] * s->dy[c] * s->dz[c];
            }
        }
    }
}

static void cons_to_prim(const Solver *s, const double U[NVAR],
                         double *rho, double *u, double *v, double *w, double *p, double *T, double *a) {
    *rho = clamp_min(U[0], 1e-10);
    *u = U[1] / *rho;
    *v = U[2] / *rho;
    *w = U[3] / *rho;
    const double ke = 0.5 * ((*u) * (*u) + (*v) * (*v) + (*w) * (*w));
    *p = (s->gamma - 1.0) * (U[4] - (*rho) * ke);
    *p = clamp_min(*p, 1e-10);
    *T = *p / *rho;
    *a = sqrt(s->gamma * (*p) / (*rho));
}

static void flux_inviscid_n(const Solver *s, const double U[NVAR], const double n[3], double Fn[NVAR]) {
    double rho, u, v, w, p, T, a;
    cons_to_prim(s, U, &rho, &u, &v, &w, &p, &T, &a);
    (void)T; (void)a;

    const double un = u * n[0] + v * n[1] + w * n[2];
    Fn[0] = rho * un;
    Fn[1] = rho * u * un + p * n[0];
    Fn[2] = rho * v * un + p * n[1];
    Fn[3] = rho * w * un + p * n[2];
    Fn[4] = (U[4] + p) * un;
}

static void numerical_flux_rusanov(const Solver *s, const double UL[NVAR], const double UR[NVAR],
                                   const double n[3], double Fhat[NVAR]) {
    double FL[NVAR], FR[NVAR];
    flux_inviscid_n(s, UL, n, FL);
    flux_inviscid_n(s, UR, n, FR);

    double rhoL, uL, vL, wL, pL, TL, aL;
    double rhoR, uR, vR, wR, pR, TR, aR;
    cons_to_prim(s, UL, &rhoL, &uL, &vL, &wL, &pL, &TL, &aL);
    cons_to_prim(s, UR, &rhoR, &uR, &vR, &wR, &pR, &TR, &aR);

    const double unL = fabs(uL * n[0] + vL * n[1] + wL * n[2]);
    const double unR = fabs(uR * n[0] + vR * n[1] + wR * n[2]);
    const double lam = fmax(unL + aL, unR + aR);

    for (int m = 0; m < NVAR; ++m) {
        Fhat[m] = 0.5 * (FL[m] + FR[m]) - 0.5 * lam * (UR[m] - UL[m]);
    }
}

static void compute_primitive_gradients(Solver *s) {
    const int ncell = s->ncell;
    double *uu = (double *)malloc((size_t)ncell * sizeof(double));
    double *vv = (double *)malloc((size_t)ncell * sizeof(double));
    double *ww = (double *)malloc((size_t)ncell * sizeof(double));
    double *TT = (double *)malloc((size_t)ncell * sizeof(double));
    if (!uu || !vv || !ww || !TT) {
        fprintf(stderr, "Gradient temporary allocation failure\n");
        free(uu); free(vv); free(ww); free(TT);
        exit(1);
    }

    for (int c = 0; c < ncell; ++c) {
        double rho, u, v, w, p, T, a;
        cons_to_prim(s, &s->u[c * NVAR], &rho, &u, &v, &w, &p, &T, &a);
        uu[c] = u;
        vv[c] = v;
        ww[c] = w;
        TT[c] = T;
    }

    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx; ++i) {
                const int c = idx3(s, i, j, k);
                const int ip = idx3(s, (i + 1) % s->nx, j, k);
                const int im = idx3(s, (i - 1 + s->nx) % s->nx, j, k);
                const int jp = idx3(s, i, (j + 1) % s->ny, k);
                const int jm = idx3(s, i, (j - 1 + s->ny) % s->ny, k);
                const int kp = idx3(s, i, j, (k + 1) % s->nz);
                const int km = idx3(s, i, j, (k - 1 + s->nz) % s->nz);

                const double dxm = 0.5 * (s->dx[c] + s->dx[im]);
                const double dxp = 0.5 * (s->dx[c] + s->dx[ip]);
                const double dym = 0.5 * (s->dy[c] + s->dy[jm]);
                const double dyp = 0.5 * (s->dy[c] + s->dy[jp]);
                const double dzm = 0.5 * (s->dz[c] + s->dz[km]);
                const double dzp = 0.5 * (s->dz[c] + s->dz[kp]);

                const double denomx = clamp_min(dxm + dxp, 1e-12);
                const double denomy = clamp_min(dym + dyp, 1e-12);
                const double denomz = clamp_min(dzm + dzp, 1e-12);

                s->gu[c * 3 + 0] = (uu[ip] - uu[im]) / denomx;
                s->gu[c * 3 + 1] = (uu[jp] - uu[jm]) / denomy;
                s->gu[c * 3 + 2] = (uu[kp] - uu[km]) / denomz;

                s->gv[c * 3 + 0] = (vv[ip] - vv[im]) / denomx;
                s->gv[c * 3 + 1] = (vv[jp] - vv[jm]) / denomy;
                s->gv[c * 3 + 2] = (vv[kp] - vv[km]) / denomz;

                s->gw[c * 3 + 0] = (ww[ip] - ww[im]) / denomx;
                s->gw[c * 3 + 1] = (ww[jp] - ww[jm]) / denomy;
                s->gw[c * 3 + 2] = (ww[kp] - ww[km]) / denomz;

                s->gT[c * 3 + 0] = (TT[ip] - TT[im]) / denomx;
                s->gT[c * 3 + 1] = (TT[jp] - TT[jm]) / denomy;
                s->gT[c * 3 + 2] = (TT[kp] - TT[km]) / denomz;
            }
        }
    }

    free(uu);
    free(vv);
    free(ww);
    free(TT);
}

static void viscous_flux_n(const Solver *s, int c, const double n[3], double Fvn[NVAR]) {
    double rho, u, v, w, p, T, a;
    cons_to_prim(s, &s->u[c * NVAR], &rho, &u, &v, &w, &p, &T, &a);
    (void)p; (void)a;

    const double du_dx = s->gu[c * 3 + 0], du_dy = s->gu[c * 3 + 1], du_dz = s->gu[c * 3 + 2];
    const double dv_dx = s->gv[c * 3 + 0], dv_dy = s->gv[c * 3 + 1], dv_dz = s->gv[c * 3 + 2];
    const double dw_dx = s->gw[c * 3 + 0], dw_dy = s->gw[c * 3 + 1], dw_dz = s->gw[c * 3 + 2];
    const double dT_dx = s->gT[c * 3 + 0], dT_dy = s->gT[c * 3 + 1], dT_dz = s->gT[c * 3 + 2];

    const double div_u = du_dx + dv_dy + dw_dz;
    const double mu = s->mu;
    const double kappa = mu * s->gamma / ((s->gamma - 1.0) * clamp_min(s->pr, 1e-12));

    const double tau_xx = mu * (2.0 * du_dx - (2.0 / 3.0) * div_u);
    const double tau_yy = mu * (2.0 * dv_dy - (2.0 / 3.0) * div_u);
    const double tau_zz = mu * (2.0 * dw_dz - (2.0 / 3.0) * div_u);
    const double tau_xy = mu * (du_dy + dv_dx);
    const double tau_xz = mu * (du_dz + dw_dx);
    const double tau_yz = mu * (dv_dz + dw_dy);

    const double qx = -kappa * dT_dx;
    const double qy = -kappa * dT_dy;
    const double qz = -kappa * dT_dz;

    const double tx_n = tau_xx * n[0] + tau_xy * n[1] + tau_xz * n[2];
    const double ty_n = tau_xy * n[0] + tau_yy * n[1] + tau_yz * n[2];
    const double tz_n = tau_xz * n[0] + tau_yz * n[1] + tau_zz * n[2];
    const double qn = qx * n[0] + qy * n[1] + qz * n[2];

    Fvn[0] = 0.0;
    Fvn[1] = tx_n;
    Fvn[2] = ty_n;
    Fvn[3] = tz_n;
    Fvn[4] = u * tx_n + v * ty_n + w * tz_n - qn;
}

static void initialize_ic(Solver *s) {
    for (int c = 0; c < s->ncell; ++c) {
        const double x = s->xc[c], y = s->yc[c], z = s->zc[c];
        const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5);

        const double rho = 1.0 + 0.2 * exp(-60.0 * r2);
        const double u = 0.2, v = 0.0, w = 0.0;
        const double p = 1.0;
        const double E = p / ((s->gamma - 1.0) * rho) + 0.5 * (u * u + v * v + w * w);

        s->u[c * NVAR + 0] = rho;
        s->u[c * NVAR + 1] = rho * u;
        s->u[c * NVAR + 2] = rho * v;
        s->u[c * NVAR + 3] = rho * w;
        s->u[c * NVAR + 4] = rho * E;
    }
}

static void add_face_flux(Solver *s, int cL, int cR, const double nL[3], double area, double hL, double hR) {
    double UL[NVAR], UR[NVAR], Fhat[NVAR], FvL[NVAR], FvR[NVAR], FvHat[NVAR];
    memcpy(UL, &s->u[cL * NVAR], sizeof(UL));
    memcpy(UR, &s->u[cR * NVAR], sizeof(UR));

    numerical_flux_rusanov(s, UL, UR, nL, Fhat);
    viscous_flux_n(s, cL, nL, FvL);
    viscous_flux_n(s, cR, nL, FvR);

    const double h = 0.5 * (hL + hR);
    const double tau = s->c_ip * s->mu / clamp_min(h, 1e-12);
    for (int m = 0; m < NVAR; ++m) {
        FvHat[m] = 0.5 * (FvL[m] + FvR[m]) - tau * (UR[m] - UL[m]);
    }
    FvHat[0] = 0.0; /* no diffusive mass flux */

    for (int m = 0; m < NVAR; ++m) {
        const double Ftotal = Fhat[m] - FvHat[m];
        s->rhs[cL * NVAR + m] -= Ftotal * area;
        s->rhs[cR * NVAR + m] += Ftotal * area;
    }
}

static void compute_rhs(Solver *s) {
    memset(s->rhs, 0, (size_t)s->ncell * NVAR * sizeof(double));
    compute_primitive_gradients(s);

    /* periodic interfaces in x-direction */
    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx; ++i) {
                int iR = (i + 1) % s->nx;
                int cL = idx3(s, i, j, k);
                int cR = idx3(s, iR, j, k);
                const double n[3] = {1.0, 0.0, 0.0};
                const double area = 0.5 * (s->dy[cL] * s->dz[cL] + s->dy[cR] * s->dz[cR]);
                add_face_flux(s, cL, cR, n, area, s->dx[cL], s->dx[cR]);
            }
        }
    }

    /* periodic interfaces in y-direction */
    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx; ++i) {
                int jR = (j + 1) % s->ny;
                int cL = idx3(s, i, j, k);
                int cR = idx3(s, i, jR, k);
                const double n[3] = {0.0, 1.0, 0.0};
                const double area = 0.5 * (s->dx[cL] * s->dz[cL] + s->dx[cR] * s->dz[cR]);
                add_face_flux(s, cL, cR, n, area, s->dy[cL], s->dy[cR]);
            }
        }
    }

    /* periodic interfaces in z-direction */
    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx; ++i) {
                int kR = (k + 1) % s->nz;
                int cL = idx3(s, i, j, k);
                int cR = idx3(s, i, j, kR);
                const double n[3] = {0.0, 0.0, 1.0};
                const double area = 0.5 * (s->dx[cL] * s->dy[cL] + s->dx[cR] * s->dy[cR]);
                add_face_flux(s, cL, cR, n, area, s->dz[cL], s->dz[cR]);
            }
        }
    }
}

static double compute_dt(const Solver *s) {
    double dt = 1e100;
    for (int c = 0; c < s->ncell; ++c) {
        double rho, u, v, w, p, T, a;
        cons_to_prim(s, &s->u[c * NVAR], &rho, &u, &v, &w, &p, &T, &a);
        const double sx = (fabs(u) + a) / s->dx[c];
        const double sy = (fabs(v) + a) / s->dy[c];
        const double sz = (fabs(w) + a) / s->dz[c];
        const double dtc = s->cfl / (sx + sy + sz + 1e-12);
        if (dtc < dt) dt = dtc;
    }
    return dt;
}

static void advance_ssprk3(Solver *s, double dt) {
    const size_t n = (size_t)s->ncell * NVAR;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    double *u2 = (double *)malloc(n * sizeof(double));
    if (!u0 || !u1 || !u2) {
        fprintf(stderr, "SSPRK3 temporary allocation failure\n");
        free(u0); free(u1); free(u2);
        exit(1);
    }

    memcpy(u0, s->u, n * sizeof(double));

    /* Stage 1: U1 = U0 + dt * L(U0) */
    compute_rhs(s);
    for (int c = 0; c < s->ncell; ++c) {
        for (int m = 0; m < NVAR; ++m) {
            const size_t q = (size_t)c * NVAR + (size_t)m;
            u1[q] = u0[q] + dt * s->rhs[q] / s->vol[c];
        }
        u1[(size_t)c * NVAR] = clamp_min(u1[(size_t)c * NVAR], 1e-8);
    }
    memcpy(s->u, u1, n * sizeof(double));

    /* Stage 2: U2 = 3/4 U0 + 1/4 (U1 + dt * L(U1)) */
    compute_rhs(s);
    for (int c = 0; c < s->ncell; ++c) {
        for (int m = 0; m < NVAR; ++m) {
            const size_t q = (size_t)c * NVAR + (size_t)m;
            const double predictor = u1[q] + dt * s->rhs[q] / s->vol[c];
            u2[q] = 0.75 * u0[q] + 0.25 * predictor;
        }
        u2[(size_t)c * NVAR] = clamp_min(u2[(size_t)c * NVAR], 1e-8);
    }
    memcpy(s->u, u2, n * sizeof(double));

    /* Stage 3: U^{n+1} = 1/3 U0 + 2/3 (U2 + dt * L(U2)) */
    compute_rhs(s);
    for (int c = 0; c < s->ncell; ++c) {
        for (int m = 0; m < NVAR; ++m) {
            const size_t q = (size_t)c * NVAR + (size_t)m;
            const double predictor = u2[q] + dt * s->rhs[q] / s->vol[c];
            s->u[q] = (1.0 / 3.0) * u0[q] + (2.0 / 3.0) * predictor;
        }
        s->u[(size_t)c * NVAR] = clamp_min(s->u[(size_t)c * NVAR], 1e-8);
    }

    free(u0);
    free(u1);
    free(u2);
}

int main(void) {
    Solver s;
    s.nx = 16;
    s.ny = 10;
    s.nz = 8;
    s.ncell = s.nx * s.ny * s.nz;
    s.gamma = 1.4;
    s.mu = 2e-4;
    s.pr = 0.72;
    s.c_ip = 2.0;
    s.cfl = 0.25;
    s.t_final = 0.2;
    s.warp = 0.08;

    alloc_solver(&s);
    build_mesh(&s);
    initialize_ic(&s);

    double t = 0.0;
    int step = 0;
    while (t < s.t_final) {
        double dt = compute_dt(&s);
        if (t + dt > s.t_final) dt = s.t_final - t;
        advance_ssprk3(&s, dt);
        t += dt;
        step++;
        if (step % 25 == 0 || t >= s.t_final) {
            printf("step=%d t=%.6f dt=%.3e\n", step, t, dt);
        }
    }

    double mass = 0.0;
    for (int c = 0; c < s.ncell; ++c) mass += s.u[c * NVAR + 0] * s.vol[c];
    printf("Done. cells=%d final_mass=%.12e\n", s.ncell, mass);

    free_solver(&s);
    return 0;
}
