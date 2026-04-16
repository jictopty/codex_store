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
    double cfl;
    double t_final;
    double warp;    /* warp amplitude for curvilinear map */

    double *xc, *yc, *zc;      /* cell centers */
    double *vol;               /* cell volumes */
    double *dx, *dy, *dz;      /* local metric lengths */

    double *u;                 /* conserved vars, size ncell*NVAR */
    double *rhs;               /* residual */
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

    if (!s->xc || !s->yc || !s->zc || !s->vol || !s->dx || !s->dy || !s->dz || !s->u || !s->rhs) {
        fprintf(stderr, "Allocation failure\n");
        exit(1);
    }
}

static void free_solver(Solver *s) {
    free(s->xc); free(s->yc); free(s->zc);
    free(s->vol); free(s->dx); free(s->dy); free(s->dz);
    free(s->u); free(s->rhs);
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

static void add_face_flux(Solver *s, int cL, int cR, const double nL[3], double area) {
    double UL[NVAR], UR[NVAR], Fhat[NVAR];
    memcpy(UL, &s->u[cL * NVAR], sizeof(UL));
    memcpy(UR, &s->u[cR * NVAR], sizeof(UR));

    numerical_flux_rusanov(s, UL, UR, nL, Fhat);

    for (int m = 0; m < NVAR; ++m) {
        s->rhs[cL * NVAR + m] -= Fhat[m] * area;
        s->rhs[cR * NVAR + m] += Fhat[m] * area;
    }
}

static void compute_rhs(Solver *s) {
    memset(s->rhs, 0, (size_t)s->ncell * NVAR * sizeof(double));

    /* periodic interfaces in x-direction */
    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx; ++i) {
                int iR = (i + 1) % s->nx;
                int cL = idx3(s, i, j, k);
                int cR = idx3(s, iR, j, k);
                const double n[3] = {1.0, 0.0, 0.0};
                const double area = 0.5 * (s->dy[cL] * s->dz[cL] + s->dy[cR] * s->dz[cR]);
                add_face_flux(s, cL, cR, n, area);
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
                add_face_flux(s, cL, cR, n, area);
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
                add_face_flux(s, cL, cR, n, area);
            }
        }
    }

    /* simple viscous regularization (Laplacian-like on momentum and energy) */
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

                const double hx2 = s->dx[c] * s->dx[c];
                const double hy2 = s->dy[c] * s->dy[c];
                const double hz2 = s->dz[c] * s->dz[c];

                for (int m = 1; m < NVAR; ++m) {
                    const double uc = s->u[c * NVAR + m];
                    const double lap = (s->u[ip * NVAR + m] - 2.0 * uc + s->u[im * NVAR + m]) / hx2
                                     + (s->u[jp * NVAR + m] - 2.0 * uc + s->u[jm * NVAR + m]) / hy2
                                     + (s->u[kp * NVAR + m] - 2.0 * uc + s->u[km * NVAR + m]) / hz2;
                    s->rhs[c * NVAR + m] += s->mu * lap * s->vol[c];
                }
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

static void advance_euler(Solver *s, double dt) {
    compute_rhs(s);
    for (int c = 0; c < s->ncell; ++c) {
        for (int m = 0; m < NVAR; ++m) {
            s->u[c * NVAR + m] += dt * s->rhs[c * NVAR + m] / s->vol[c];
        }
        s->u[c * NVAR + 0] = clamp_min(s->u[c * NVAR + 0], 1e-8);
    }
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
        advance_euler(&s, dt);
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
