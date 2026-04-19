#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MPI
#include <mpi.h>
#else
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Status;
#define MPI_COMM_WORLD 0
#define MPI_DOUBLE 0
#define MPI_MIN 0
#define MPI_SUM 1
#define MPI_STATUS_IGNORE ((MPI_Status *)0)
static int MPI_Init(void *a, void *b) { (void)a; (void)b; return 0; }
static int MPI_Finalize(void) { return 0; }
static int MPI_Comm_rank(MPI_Comm c, int *r) { (void)c; *r = 0; return 0; }
static int MPI_Comm_size(MPI_Comm c, int *n) { (void)c; *n = 1; return 0; }
static int MPI_Allreduce(const double *s, double *r, int n, MPI_Datatype t, MPI_Op op, MPI_Comm c) {
    (void)t; (void)op; (void)c;
    for (int i = 0; i < n; ++i) r[i] = s[i];
    return 0;
}
static int MPI_Sendrecv(const double *sb, int sc, MPI_Datatype st, int d, int stag,
                        double *rb, int rc, MPI_Datatype rt, int src, int rtag,
                        MPI_Comm c, MPI_Status *stat) {
    (void)st; (void)d; (void)stag; (void)rt; (void)src; (void)rtag; (void)c; (void)stat;
    const int n = sc < rc ? sc : rc;
    for (int i = 0; i < n; ++i) rb[i] = sb[i];
    return 0;
}
static int MPI_Abort(MPI_Comm c, int e) { (void)c; exit(e); }
#endif

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
    int nx, ny, nz, ncell;      /* local dimensions */
    int gnx;                    /* global x-cells */
    int i_start;                /* global x offset for this rank */
    int rank, nranks;
    int p_order;                /* requested polynomial order */
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
    double *prho, *pu, *pv, *pw, *pT; /* primitive fields at cell centers */
} Solver;

static inline int idx3(const Solver *s, int i, int j, int k) {
    return (k * s->ny + j) * s->nx + i;
}

static inline double clamp_min(double x, double xmin) {
    return x < xmin ? xmin : x;
}

static void decompose_x(int gnx, int nranks, int rank, int *nx_local, int *i_start) {
    const int base = gnx / nranks;
    const int rem = gnx % nranks;
    *nx_local = base + (rank < rem ? 1 : 0);
    *i_start = rank * base + (rank < rem ? rank : rem);
}

static void exchange_x_plane_scalar(const Solver *s, const double *field, double *left_plane, double *right_plane) {
    const int left = (s->rank - 1 + s->nranks) % s->nranks;
    const int right = (s->rank + 1) % s->nranks;
    const int plane_n = s->ny * s->nz;
    double *send_left = (double *)malloc((size_t)plane_n * sizeof(double));
    double *send_right = (double *)malloc((size_t)plane_n * sizeof(double));
    if (!send_left || !send_right) {
        fprintf(stderr, "MPI plane allocation failure\n");
        free(send_left); free(send_right);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int q = 0;
    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            send_left[q] = field[idx3(s, 0, j, k)];
            send_right[q] = field[idx3(s, s->nx - 1, j, k)];
            q++;
        }
    }

    MPI_Sendrecv(send_left, plane_n, MPI_DOUBLE, left, 10,
                 right_plane, plane_n, MPI_DOUBLE, right, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_right, plane_n, MPI_DOUBLE, right, 20,
                 left_plane, plane_n, MPI_DOUBLE, left, 20,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    free(send_left);
    free(send_right);
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
    s->prho = (double *)calloc(n, sizeof(double));
    s->pu = (double *)calloc(n, sizeof(double));
    s->pv = (double *)calloc(n, sizeof(double));
    s->pw = (double *)calloc(n, sizeof(double));
    s->pT = (double *)calloc(n, sizeof(double));

    if (!s->xc || !s->yc || !s->zc || !s->vol || !s->dx || !s->dy || !s->dz ||
        !s->u || !s->rhs || !s->gu || !s->gv || !s->gw || !s->gT ||
        !s->prho || !s->pu || !s->pv || !s->pw || !s->pT) {
        fprintf(stderr, "Allocation failure\n");
        exit(1);
    }
}

static void free_solver(Solver *s) {
    free(s->xc); free(s->yc); free(s->zc);
    free(s->vol); free(s->dx); free(s->dy); free(s->dz);
    free(s->u); free(s->rhs);
    free(s->gu); free(s->gv); free(s->gw); free(s->gT);
    free(s->prho); free(s->pu); free(s->pv); free(s->pw); free(s->pT);
}

static void build_mesh(Solver *s) {
    const double dxi = 1.0 / s->gnx;
    const double deta = 1.0 / s->ny;
    const double dzeta = 1.0 / s->nz;

    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx; ++i) {
                const int c = idx3(s, i, j, k);

                const int gi = s->i_start + i;
                const double xi = (gi + 0.5) * dxi;
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

static void prim_to_cons(const Solver *s, double rho, double u, double v, double w, double p, double U[NVAR]) {
    rho = clamp_min(rho, 1e-10);
    p = clamp_min(p, 1e-10);
    const double E = p / ((s->gamma - 1.0) * rho) + 0.5 * (u * u + v * v + w * w);
    U[0] = rho;
    U[1] = rho * u;
    U[2] = rho * v;
    U[3] = rho * w;
    U[4] = rho * E;
}

static void reconstruct_face_state(const Solver *s, int c, const double n[3], double h, int side, double Uface[NVAR]) {
    int i = c % s->nx;
    int j = (c / s->nx) % s->ny;
    int k = c / (s->nx * s->ny);

    int dir = 0;
    if (fabs(n[1]) > fabs(n[dir])) dir = 1;
    if (fabs(n[2]) > fabs(n[dir])) dir = 2;

    int p_eff = s->p_order;
    if (s->nranks > 1 && dir == 0 && p_eff > 1) {
        p_eff = 1; /* MPI path only exchanges one halo plane in x */
    }

    if (p_eff <= 0) {
        const double rho = s->prho[c];
        const double u = s->pu[c], v = s->pv[c], w = s->pw[c], T = s->pT[c];
        const double p = clamp_min(rho * T, 1e-10);
        prim_to_cons(s, rho, u, v, w, p, Uface);
        return;
    }

    if (p_eff == 1) {
        const double rho = s->prho[c];
        const double u = s->pu[c], v = s->pv[c], w = s->pw[c], T = s->pT[c];
        const double ds = 0.5 * h * (double)side;
        const double du = (s->gu[c * 3 + 0] * n[0] + s->gu[c * 3 + 1] * n[1] + s->gu[c * 3 + 2] * n[2]) * ds;
        const double dv = (s->gv[c * 3 + 0] * n[0] + s->gv[c * 3 + 1] * n[1] + s->gv[c * 3 + 2] * n[2]) * ds;
        const double dw = (s->gw[c * 3 + 0] * n[0] + s->gw[c * 3 + 1] * n[1] + s->gw[c * 3 + 2] * n[2]) * ds;
        const double dT = (s->gT[c * 3 + 0] * n[0] + s->gT[c * 3 + 1] * n[1] + s->gT[c * 3 + 2] * n[2]) * ds;
        const double p = clamp_min(rho * clamp_min(T + dT, 1e-10), 1e-10);
        prim_to_cons(s, rho, u + du, v + dv, w + dw, p, Uface);
        return;
    }

    /* arbitrary-order (p>=2) face interpolation from cell-centered primitive values */
    const double x_eval = 0.5 * (double)side;
    const int npt = p_eff + 1;
    const int start = -(npt / 2);
    double rho_f = 0.0, u_f = 0.0, v_f = 0.0, w_f = 0.0, T_f = 0.0;
    for (int a = 0; a < npt; ++a) {
        const int off_a = start + a;
        double xa = (double)off_a;
        double la = 1.0;
        for (int b = 0; b < npt; ++b) {
            if (a == b) continue;
            const int off_b = start + b;
            const double xb = (double)off_b;
            la *= (x_eval - xb) / (xa - xb);
        }

        int ii = i, jj = j, kk = k;
        if (dir == 0) ii = (i + off_a + s->nx) % s->nx;
        if (dir == 1) jj = (j + off_a + s->ny) % s->ny;
        if (dir == 2) kk = (k + off_a + s->nz) % s->nz;
        const int cs = idx3(s, ii, jj, kk);

        rho_f += la * s->prho[cs];
        u_f += la * s->pu[cs];
        v_f += la * s->pv[cs];
        w_f += la * s->pw[cs];
        T_f += la * s->pT[cs];
    }

    rho_f = clamp_min(rho_f, 1e-10);
    T_f = clamp_min(T_f, 1e-10);
    prim_to_cons(s, rho_f, u_f, v_f, w_f, clamp_min(rho_f * T_f, 1e-10), Uface);
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
    double *uu = s->pu;
    double *vv = s->pv;
    double *ww = s->pw;
    double *TT = s->pT;
    double *rr = s->prho;

    for (int c = 0; c < ncell; ++c) {
        double rho, u, v, w, p, T, a;
        cons_to_prim(s, &s->u[c * NVAR], &rho, &u, &v, &w, &p, &T, &a);
        rr[c] = rho;
        uu[c] = u;
        vv[c] = v;
        ww[c] = w;
        TT[c] = T;
    }

    const int plane_n = s->ny * s->nz;
    double *uu_left = (double *)malloc((size_t)plane_n * sizeof(double));
    double *uu_right = (double *)malloc((size_t)plane_n * sizeof(double));
    double *vv_left = (double *)malloc((size_t)plane_n * sizeof(double));
    double *vv_right = (double *)malloc((size_t)plane_n * sizeof(double));
    double *ww_left = (double *)malloc((size_t)plane_n * sizeof(double));
    double *ww_right = (double *)malloc((size_t)plane_n * sizeof(double));
    double *TT_left = (double *)malloc((size_t)plane_n * sizeof(double));
    double *TT_right = (double *)malloc((size_t)plane_n * sizeof(double));
    if (!uu_left || !uu_right || !vv_left || !vv_right || !ww_left || !ww_right || !TT_left || !TT_right) {
        fprintf(stderr, "Gradient halo allocation failure\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    exchange_x_plane_scalar(s, uu, uu_left, uu_right);
    exchange_x_plane_scalar(s, vv, vv_left, vv_right);
    exchange_x_plane_scalar(s, ww, ww_left, ww_right);
    exchange_x_plane_scalar(s, TT, TT_left, TT_right);

    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx; ++i) {
                const int c = idx3(s, i, j, k);
                const int ip = (i + 1 < s->nx) ? idx3(s, i + 1, j, k) : -1;
                const int im = (i - 1 >= 0) ? idx3(s, i - 1, j, k) : -1;
                const int jp = idx3(s, i, (j + 1) % s->ny, k);
                const int jm = idx3(s, i, (j - 1 + s->ny) % s->ny, k);
                const int kp = idx3(s, i, j, (k + 1) % s->nz);
                const int km = idx3(s, i, j, (k - 1 + s->nz) % s->nz);
                const int plane_id = k * s->ny + j;

                const double dxm = 0.5 * (s->dx[c] + (im >= 0 ? s->dx[im] : s->dx[c]));
                const double dxp = 0.5 * (s->dx[c] + (ip >= 0 ? s->dx[ip] : s->dx[c]));
                const double dym = 0.5 * (s->dy[c] + s->dy[jm]);
                const double dyp = 0.5 * (s->dy[c] + s->dy[jp]);
                const double dzm = 0.5 * (s->dz[c] + s->dz[km]);
                const double dzp = 0.5 * (s->dz[c] + s->dz[kp]);

                const double denomx = clamp_min(dxm + dxp, 1e-12);
                const double denomy = clamp_min(dym + dyp, 1e-12);
                const double denomz = clamp_min(dzm + dzp, 1e-12);

                const double uu_ip = (ip >= 0) ? uu[ip] : uu_right[plane_id];
                const double uu_im = (im >= 0) ? uu[im] : uu_left[plane_id];
                const double vv_ip = (ip >= 0) ? vv[ip] : vv_right[plane_id];
                const double vv_im = (im >= 0) ? vv[im] : vv_left[plane_id];
                const double ww_ip = (ip >= 0) ? ww[ip] : ww_right[plane_id];
                const double ww_im = (im >= 0) ? ww[im] : ww_left[plane_id];
                const double TT_ip = (ip >= 0) ? TT[ip] : TT_right[plane_id];
                const double TT_im = (im >= 0) ? TT[im] : TT_left[plane_id];

                s->gu[c * 3 + 0] = (uu_ip - uu_im) / denomx;
                s->gu[c * 3 + 1] = (uu[jp] - uu[jm]) / denomy;
                s->gu[c * 3 + 2] = (uu[kp] - uu[km]) / denomz;

                s->gv[c * 3 + 0] = (vv_ip - vv_im) / denomx;
                s->gv[c * 3 + 1] = (vv[jp] - vv[jm]) / denomy;
                s->gv[c * 3 + 2] = (vv[kp] - vv[km]) / denomz;

                s->gw[c * 3 + 0] = (ww_ip - ww_im) / denomx;
                s->gw[c * 3 + 1] = (ww[jp] - ww[jm]) / denomy;
                s->gw[c * 3 + 2] = (ww[kp] - ww[km]) / denomz;

                s->gT[c * 3 + 0] = (TT_ip - TT_im) / denomx;
                s->gT[c * 3 + 1] = (TT[jp] - TT[jm]) / denomy;
                s->gT[c * 3 + 2] = (TT[kp] - TT[km]) / denomz;
            }
        }
    }

    free(uu_left); free(uu_right);
    free(vv_left); free(vv_right);
    free(ww_left); free(ww_right);
    free(TT_left); free(TT_right);
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
    reconstruct_face_state(s, cL, nL, hL, +1, UL);
    reconstruct_face_state(s, cR, nL, hR, -1, UR);

    numerical_flux_rusanov(s, UL, UR, nL, Fhat);
    viscous_flux_n(s, cL, nL, FvL);
    viscous_flux_n(s, cR, nL, FvR);

    const double h = 0.5 * (hL + hR);
    const double pscale = (double)(s->p_order + 1) * (double)(s->p_order + 1);
    const double tau = s->c_ip * pscale * s->mu / clamp_min(h, 1e-12);
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

static void add_face_flux_remote(Solver *s, int cL, const double UR[NVAR], const double FvR[NVAR],
                                 const double nL[3], double area, double hL, double hR) {
    double UL[NVAR], Fhat[NVAR], FvL[NVAR], FvHat[NVAR];
    reconstruct_face_state(s, cL, nL, hL, +1, UL);
    numerical_flux_rusanov(s, UL, UR, nL, Fhat);
    viscous_flux_n(s, cL, nL, FvL);

    const double h = 0.5 * (hL + hR);
    const double pscale = (double)(s->p_order + 1) * (double)(s->p_order + 1);
    const double tau = s->c_ip * pscale * s->mu / clamp_min(h, 1e-12);
    for (int m = 0; m < NVAR; ++m) {
        FvHat[m] = 0.5 * (FvL[m] + FvR[m]) - tau * (UR[m] - UL[m]);
    }
    FvHat[0] = 0.0;

    for (int m = 0; m < NVAR; ++m) {
        const double Ftotal = Fhat[m] - FvHat[m];
        s->rhs[cL * NVAR + m] -= Ftotal * area;
    }
}

static void compute_rhs(Solver *s) {
    memset(s->rhs, 0, (size_t)s->ncell * NVAR * sizeof(double));
    compute_primitive_gradients(s);

    /* x-direction interfaces: local interior + right MPI boundary */
    for (int k = 0; k < s->nz; ++k) {
        for (int j = 0; j < s->ny; ++j) {
            for (int i = 0; i < s->nx - 1; ++i) {
                int iR = i + 1;
                int cL = idx3(s, i, j, k);
                int cR = idx3(s, iR, j, k);
                const double n[3] = {1.0, 0.0, 0.0};
                const double area = 0.5 * (s->dy[cL] * s->dz[cL] + s->dy[cR] * s->dz[cR]);
                add_face_flux(s, cL, cR, n, area, s->dx[cL], s->dx[cR]);
            }
        }
    }

    {
        const int plane_n = s->ny * s->nz;
        double *u_right = (double *)malloc((size_t)plane_n * NVAR * sizeof(double));
        double *send_left_face = (double *)malloc((size_t)plane_n * NVAR * sizeof(double));
        double *fv_left = (double *)malloc((size_t)plane_n * NVAR * sizeof(double));
        double *fv_right = (double *)malloc((size_t)plane_n * NVAR * sizeof(double));
        if (!u_right || !send_left_face || !fv_left || !fv_right) {
            fprintf(stderr, "Boundary plane allocation failure\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int q = 0;
        for (int k = 0; k < s->nz; ++k) {
            for (int j = 0; j < s->ny; ++j) {
                const int cL = idx3(s, 0, j, k);
                const int cR = idx3(s, s->nx - 1, j, k);
                const double n[3] = {1.0, 0.0, 0.0};
                reconstruct_face_state(s, cL, n, s->dx[cL], -1, &send_left_face[q]);
                viscous_flux_n(s, cL, n, &fv_left[q]);
                viscous_flux_n(s, cR, n, &fv_right[q]);
                q += NVAR;
            }
        }

        const int left = (s->rank - 1 + s->nranks) % s->nranks;
        const int right = (s->rank + 1) % s->nranks;
        (void)left; (void)right;
        MPI_Sendrecv(send_left_face, plane_n * NVAR, MPI_DOUBLE, left, 45,
                     u_right, plane_n * NVAR, MPI_DOUBLE, right, 45,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(fv_left, plane_n * NVAR, MPI_DOUBLE, left, 50,
                     fv_right, plane_n * NVAR, MPI_DOUBLE, right, 50,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        q = 0;
        for (int k = 0; k < s->nz; ++k) {
            for (int j = 0; j < s->ny; ++j) {
                const int cL = idx3(s, s->nx - 1, j, k);
                const double n[3] = {1.0, 0.0, 0.0};
                const double area = s->dy[cL] * s->dz[cL];
                add_face_flux_remote(s, cL, &u_right[q], &fv_right[q], n, area, s->dx[cL], s->dx[cL]);
                q += NVAR;
            }
        }

        free(u_right);
        free(send_left_face);
        free(fv_left);
        free(fv_right);
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
    double dt_local = 1e100;
    for (int c = 0; c < s->ncell; ++c) {
        double rho, u, v, w, p, T, a;
        cons_to_prim(s, &s->u[c * NVAR], &rho, &u, &v, &w, &p, &T, &a);
        const double sx = (fabs(u) + a) / s->dx[c];
        const double sy = (fabs(v) + a) / s->dy[c];
        const double sz = (fabs(w) + a) / s->dz[c];
        const double pscale = 2.0 * (double)s->p_order + 1.0;
        const double dtc = s->cfl / (pscale * (sx + sy + sz + 1e-12));
        if (dtc < dt_local) dt_local = dtc;
    }
    double dt = dt_local;
    MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
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

int main(int argc, char **argv) {
    MPI_Init(NULL, NULL);

    Solver s;
    MPI_Comm_rank(MPI_COMM_WORLD, &s.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s.nranks);

    s.gnx = 16;
    s.ny = 10;
    s.nz = 8;
    decompose_x(s.gnx, s.nranks, s.rank, &s.nx, &s.i_start);
    if (s.nx <= 0) {
        if (s.rank == 0) {
            fprintf(stderr, "Error: number of MPI ranks exceeds global x-cells.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    s.ncell = s.nx * s.ny * s.nz;
    s.gamma = 1.4;
    s.mu = 2e-4;
    s.pr = 0.72;
    s.c_ip = 2.0;
    s.p_order = 0;
    if (argc > 1) {
        s.p_order = atoi(argv[1]);
    }
    if (s.p_order < 0) s.p_order = 0;
    if (s.rank == 0 && s.nranks > 1 && s.p_order > 1) {
        printf("Note: with MPI x-decomposition, p>1 in x-direction currently falls back to first-order x-face reconstruction.\n");
    }
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
        if ((step % 25 == 0 || t >= s.t_final) && s.rank == 0) {
            printf("step=%d t=%.6f dt=%.3e p=%d\n", step, t, dt, s.p_order);
        }
    }

    double mass_local = 0.0;
    for (int c = 0; c < s.ncell; ++c) mass_local += s.u[c * NVAR + 0] * s.vol[c];
    double mass = 0.0;
    MPI_Allreduce(&mass_local, &mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (s.rank == 0) {
        printf("Done. local_cells=%d mpi_ranks=%d p=%d final_mass=%.12e\n", s.ncell, s.nranks, s.p_order, mass);
    }

    free_solver(&s);
    MPI_Finalize();
    return 0;
}
